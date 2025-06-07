from __future__ import annotations
import os, json, re, hashlib, uuid, logging, time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()

from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from qdrant_client import QdrantClient, models
from langchain_qdrant import QdrantVectorStore
from langchain_aws import BedrockEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Importaciones para procesamiento de PDF
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("⚠️ PyMuPDF no disponible, usando PyPDF2 como respaldo")

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    print("❌ PyPDF2 no disponible")

# ---------- 1. Logging Setup -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('educational_pdf_processing.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------- 2. Cost Tracking -------------------------------------------------
class CostTracker:
    def __init__(self):
        self.embedding_calls = 0
        self.embedding_tokens = 0
        self.batch_count = 0
        self.titan_embed_price = 0.0001  # $0.0001 per 1K tokens
    
    def add_embedding_batch(self, documents: List[Document]):
        """Trackea costos de un batch de embeddings"""
        self.batch_count += 1
        total_text_length = sum(len(doc.page_content) for doc in documents)
        estimated_tokens = max(len(documents), total_text_length // 4)
        
        self.embedding_calls += len(documents)
        self.embedding_tokens += estimated_tokens
        
        logger.info(f"📊 Batch {self.batch_count}: {len(documents)} docs, ~{estimated_tokens:,} tokens")
    
    def get_costs(self) -> dict:
        """Calcula costos totales"""
        total_cost = (self.embedding_tokens / 1000) * self.titan_embed_price
        return {
            "embedding_calls": self.embedding_calls,
            "embedding_tokens": self.embedding_tokens,
            "batches_processed": self.batch_count,
            "cost_per_1k_tokens": self.titan_embed_price,
            "total_cost_usd": round(total_cost, 6)
        }

cost_tracker = CostTracker()

# ---------- 3. Processing Stats ----------------------------------------------
@dataclass
class ProcessingStats:
    total_docs: int = 0
    total_chunks: int = 0
    new_chunks: int = 0
    duplicate_chunks: int = 0
    processing_time: float = 0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []

# ---------- 4. Configuration -------------------------------------------------
class Settings(BaseSettings):
    pdf_base_path: Path = Path("Agentes/pdf")
    region: str = "us-east-1"
    profile: str = "832257724409_DevAnalitica"
    qdrant_url: str = "http://localhost:6333"
    collection: str = "material_educativo"
    batch_size: int = 32
    chunk_size: int = 1500
    chunk_overlap: int = 200
    min_chunk_length: int = 50
    max_chunk_length: int = 3000
    embedding_dimensions: int = 1536
    
    class Config:
        env_file = ".env"

# ---------- 5. PDF Processor -------------------------------------------------
class PDFProcessor:
    """Procesador especializado para PDFs educativos"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def extract_text_pymupdf(self, pdf_path: Path) -> str:
        """Extrae texto usando PyMuPDF (mejor calidad)"""
        if not PYMUPDF_AVAILABLE:
            return self.extract_text_pypdf2(pdf_path)
            
        try:
            doc = fitz.open(str(pdf_path))
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            logger.warning(f"PyMuPDF falló para {pdf_path}: {e}")
            return self.extract_text_pypdf2(pdf_path)
    
    def extract_text_pypdf2(self, pdf_path: Path) -> str:
        """Fallback usando PyPDF2"""
        if not PYPDF2_AVAILABLE:
            logger.error("Ninguna librería de PDF disponible")
            return ""
            
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                return text
        except Exception as e:
            logger.error(f"Error extrayendo texto de {pdf_path}: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Limpia y normaliza el texto extraído"""
        if not text:
            return ""
            
        # Normalizar espacios en blanco
        text = re.sub(r'\s+', ' ', text)
        # Remover caracteres especiales problemáticos pero mantener tildes
        text = re.sub(r'[^\w\s\.,;:!?¡¿\-\(\)\"\'áéíóúñÁÉÍÓÚÑüÜ]', '', text)
        # Normalizar puntuación
        text = re.sub(r'\.{2,}', '.', text)
        return text.strip()
    
    def extract_title_from_text(self, text: str) -> str:
        """Extrae un título probable del texto"""
        lines = text.split('\n')[:10]  # Primeras 10 líneas
        for line in lines:
            line = line.strip()
            if len(line) > 10 and len(line) < 100:  # Longitud típica de título
                return line
        return "Sin título"
    
    def process_pdf(self, pdf_path: Path, course_info: Dict[str, str]) -> List[Document]:
        """Procesa un PDF individual y retorna documentos chunkeados"""
        logger.info(f"📄 Procesando: {pdf_path.name}")
        
        # Extraer texto
        raw_text = self.extract_text_pymupdf(pdf_path)
        if not raw_text.strip():
            logger.warning(f"No se pudo extraer texto de {pdf_path}")
            return []
        
        # Limpiar texto
        clean_text = self.clean_text(raw_text)
        if len(clean_text) < self.settings.min_chunk_length:
            logger.warning(f"Texto muy corto en {pdf_path}: {len(clean_text)} caracteres")
            return []
        
        # Extraer título
        title = self.extract_title_from_text(clean_text)
        
        # Crear documento inicial
        doc = Document(
            page_content=clean_text,
            metadata={
                **course_info,
                "source_file": str(pdf_path),
                "file_name": pdf_path.name,
                "file_stem": pdf_path.stem,
                "file_size": pdf_path.stat().st_size,
                "title": title,
                "processed_at": datetime.now().isoformat(),
                "content_type": "educational_material",
                "text_length": len(clean_text)
            }
        )
        
        # Dividir en chunks
        chunks = self.text_splitter.split_documents([doc])
        
        # Agregar metadata específica a cada chunk
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_id": f"{pdf_path.stem}_{i}",
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_hash": hashlib.md5(chunk.page_content.encode()).hexdigest(),
                "chunk_length": len(chunk.page_content)
            })
        
        # Filtrar chunks válidos
        valid_chunks = [
            chunk for chunk in chunks 
            if len(chunk.page_content) >= self.settings.min_chunk_length
        ]
        
        logger.info(f"✅ {pdf_path.name}: {len(valid_chunks)} chunks válidos de {len(chunks)} totales")
        return valid_chunks

# ---------- 6. Course Scanner -----------------------------------------------
class CourseScanner:
    """Escanea la estructura de cursos en el sistema de archivos"""
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
    
    def scan_courses(self) -> Dict[str, List[Path]]:
        """Escanea y organiza los PDFs por curso"""
        courses = {}
        
        if not self.base_path.exists():
            logger.error(f"Ruta base no existe: {self.base_path}")
            return courses
        
        # Buscar PDFs recursivamente
        pdf_files = list(self.base_path.rglob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No se encontraron archivos PDF en {self.base_path}")
            return courses
        
        for pdf_path in pdf_files:
            # Determinar el curso basado en la estructura de carpetas
            relative_path = pdf_path.relative_to(self.base_path)
            course_parts = relative_path.parts[:-1]  # Todas las partes excepto el archivo
            
            if course_parts:
                course_name = "/".join(course_parts)
            else:
                course_name = "general"
            
            if course_name not in courses:
                courses[course_name] = []
            courses[course_name].append(pdf_path)
        
        logger.info(f"📚 Cursos encontrados: {list(courses.keys())}")
        for course, files in courses.items():
            logger.info(f"  - {course}: {len(files)} archivos PDF")
        
        return courses

class QdrantEducationalManager:
    """Gestor principal para el material educativo en Qdrant"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = QdrantClient(url=settings.qdrant_url)
        self.embeddings = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v1",
            region_name=settings.region,
            credentials_profile_name=settings.profile
        )
        self.pdf_processor = PDFProcessor(settings)
        self.course_scanner = CourseScanner(settings.pdf_base_path)
    
    def ensure_collection(self):
        """Crea la colección si no existe"""
        try:
            collection_info = self.client.get_collection(self.settings.collection)
            logger.info(f"✅ Colección '{self.settings.collection}' ya existe")
        except Exception:
            logger.info(f"🔨 Creando colección '{self.settings.collection}'")
            self.client.create_collection(
                collection_name=self.settings.collection,
                vectors_config=models.VectorParams(
                    size=self.settings.embedding_dimensions,
                    distance=models.Distance.COSINE
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    default_segment_number=2,
                    memmap_threshold=10000
                ),
                hnsw_config=models.HnswConfigDiff(
                    m=16,
                    ef_construct=100,
                    full_scan_threshold=10000
                )
            )
    
    def get_existing_chunk_hashes(self) -> set:
        """Obtiene hashes de chunks existentes para evitar duplicados"""
        try:
            search_result = self.client.scroll(
                collection_name=self.settings.collection,
                limit=10000,
                with_payload=True,
                with_vectors=False
            )
            
            existing_hashes = set()
            for point in search_result[0]:
                if point.payload and 'chunk_hash' in point.payload:
                    existing_hashes.add(point.payload['chunk_hash'])
            
            logger.info(f"📋 {len(existing_hashes)} chunks existentes encontrados")
            return existing_hashes
            
        except Exception as e:
            logger.warning(f"Error obteniendo chunks existentes: {e}")
            return set()
    
    def determine_course_level(self, course_name: str) -> str:
        """Determina el nivel educativo basado en el nombre del curso"""
        course_lower = course_name.lower()
        level_mapping = {
            'primaria': ['primaria', 'elementary', 'básico', 'primer', 'segundo', 'tercero', 'cuarto', 'quinto', 'sexto'],
            'secundaria': ['secundaria', 'high school', 'medio', 'séptimo', 'octavo', 'noveno', 'décimo', 'undécimo'],
            'universitario': ['universidad', 'university', 'superior', 'college', 'pregrado', 'postgrado'],
            'preescolar': ['preescolar', 'jardín', 'kinder', 'inicial']
        }
        
        for level, keywords in level_mapping.items():
            if any(keyword in course_lower for keyword in keywords):
                return level
        
        return 'general'
    
    def determine_subject_area(self, course_name: str) -> str:
        """Determina el área temática basada en el nombre del curso"""
        course_lower = course_name.lower()
        subject_mapping = {
            'matematicas': ['matemáticas', 'math', 'algebra', 'geometria', 'calculo', 'aritmetica', 'trigonometria'],
            'historia': ['historia', 'history', 'civica', 'social', 'geografía', 'democracia'],
            'ciencias': ['ciencias', 'science', 'biologia', 'fisica', 'quimica', 'naturales', 'laboratorio'],
            'lengua': ['lengua', 'language', 'literatura', 'español', 'english', 'comunicación', 'lectura'],
            'arte': ['arte', 'art', 'música', 'music', 'dibujo', 'pintura', 'cultural'],
            'tecnologia': ['tecnología', 'informática', 'computación', 'sistemas', 'programación'],
            'educacion_fisica': ['educación física', 'deporte', 'gimnasia', 'recreación']
        }
        
        for subject, keywords in subject_mapping.items():
            if any(keyword in course_lower for keyword in keywords):
                return subject
        
        return 'general'
    
    def process_course_batch(self, course_name: str, pdf_files: List[Path]) -> ProcessingStats:
        """Procesa un lote de PDFs de un curso"""
        stats = ProcessingStats()
        start_time = time.time()
        
        logger.info(f"🎓 Procesando curso: {course_name} ({len(pdf_files)} archivos)")
        
        # Información del curso
        course_info = {
            "course_name": course_name,
            "course_level": self.determine_course_level(course_name),
            "subject_area": self.determine_subject_area(course_name),
            "course_path": course_name.replace("/", "_")
        }
        
        all_documents = []
        
        # Procesar cada PDF
        for pdf_path in pdf_files:
            try:
                documents = self.pdf_processor.process_pdf(pdf_path, course_info)
                all_documents.extend(documents)
                stats.total_docs += 1
            except Exception as e:
                error_msg = f"Error procesando {pdf_path}: {e}"
                logger.error(error_msg)
                stats.errors.append(error_msg)
        
        stats.total_chunks = len(all_documents)
        
        if all_documents:
            # Filtrar duplicados
            existing_hashes = self.get_existing_chunk_hashes()
            new_documents = [
                doc for doc in all_documents 
                if doc.metadata.get('chunk_hash') not in existing_hashes
            ]
            
            stats.new_chunks = len(new_documents)
            stats.duplicate_chunks = len(all_documents) - len(new_documents)
            
            if new_documents:
                # Subir a Qdrant
                self.upload_documents_batch(new_documents)
                logger.info(f"✅ {course_name}: {stats.new_chunks} chunks nuevos subidos")
            else:
                logger.info(f"⚠️ {course_name}: No hay documentos nuevos")
        else:
            logger.warning(f"❌ {course_name}: No se generaron documentos válidos")
        
        stats.processing_time = time.time() - start_time
        return stats
    
    def upload_documents_batch(self, documents: List[Document]):
        """Sube documentos a Qdrant en batches"""
        logger.info(f"⬆️ Subiendo {len(documents)} documentos...")
        
        # Crear vector store
        vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.settings.collection,
            embeddings=self.embeddings
        )
        
        # Procesar en batches
        for i in range(0, len(documents), self.settings.batch_size):
            batch = documents[i:i + self.settings.batch_size]
            
            # Trackear costos
            cost_tracker.add_embedding_batch(batch)
            
            try:
                # Generar IDs únicos
                batch_ids = [str(uuid.uuid4()) for _ in batch]
                
                # Subir batch
                vector_store.add_documents(batch, ids=batch_ids)
                
                logger.info(f"✅ Batch {i//self.settings.batch_size + 1}: {len(batch)} documentos subidos")
                
            except Exception as e:
                logger.error(f"❌ Error en batch {i//self.settings.batch_size + 1}: {e}")
                raise
    
    def process_all_courses(self) -> Dict[str, ProcessingStats]:
        """Procesa todos los cursos encontrados"""
        logger.info("🚀 Iniciando procesamiento de material educativo")
        
        # Validar configuración
        if not self.settings.pdf_base_path.exists():
            logger.error(f"❌ Ruta de PDFs no existe: {self.settings.pdf_base_path}")
            return {}
        
        # Asegurar colección
        try:
            self.ensure_collection()
        except Exception as e:
            logger.error(f"❌ Error configurando Qdrant: {e}")
            return {}
        
        # Escanear cursos
        courses = self.course_scanner.scan_courses()
        
        if not courses:
            logger.warning("⚠️ No se encontraron cursos para procesar")
            return {}
        
        # Procesar cada curso
        results = {}
        total_start_time = time.time()
        
        for course_name, pdf_files in courses.items():
            try:
                stats = self.process_course_batch(course_name, pdf_files)
                results[course_name] = stats
                
                logger.info(f"📊 {course_name}: {stats.new_chunks} nuevos, {stats.duplicate_chunks} duplicados, {len(stats.errors)} errores")
                
            except Exception as e:
                logger.error(f"❌ Error procesando curso {course_name}: {e}")
                results[course_name] = ProcessingStats(errors=[str(e)])
        
        # Resumen final
        total_time = time.time() - total_start_time
        self.print_final_summary(results, total_time)
        
        return results
    
    def print_final_summary(self, results: Dict[str, ProcessingStats], total_time: float):
        """Imprime resumen final del procesamiento"""
        total_stats = {
            'courses_processed': len(results),
            'total_docs': sum(stats.total_docs for stats in results.values()),
            'total_chunks': sum(stats.total_chunks for stats in results.values()),
            'new_chunks': sum(stats.new_chunks for stats in results.values()),
            'duplicate_chunks': sum(stats.duplicate_chunks for stats in results.values()),
            'total_errors': sum(len(stats.errors) for stats in results.values())
        }
        
        # Costos
        costs = cost_tracker.get_costs()
        
        logger.info("=" * 80)
        logger.info("📋 RESUMEN FINAL DEL PROCESAMIENTO")
        logger.info("=" * 80)
        logger.info(f"🎓 Cursos procesados: {total_stats['courses_processed']}")
        logger.info(f"📄 PDFs procesados: {total_stats['total_docs']}")
        logger.info(f"📝 Chunks totales generados: {total_stats['total_chunks']}")
        logger.info(f"✅ Chunks nuevos subidos: {total_stats['new_chunks']}")
        logger.info(f"🔄 Chunks duplicados omitidos: {total_stats['duplicate_chunks']}")
        logger.info(f"❌ Errores encontrados: {total_stats['total_errors']}")
        logger.info(f"⏱️ Tiempo total: {total_time:.2f} segundos")
        logger.info("-" * 40)
        logger.info("💰 INFORMACIÓN DE COSTOS:")
        logger.info(f"📊 Llamadas de embedding: {costs['embedding_calls']:,}")
        logger.info(f"🔤 Tokens procesados: {costs['embedding_tokens']:,}")
        logger.info(f"💵 Costo estimado: ${costs['total_cost_usd']:.6f} USD")
        logger.info("=" * 80)
        
        # Detalles por curso
        if results:
            logger.info("📚 DETALLES POR CURSO:")
            for course_name, stats in results.items():
                logger.info(f"  - {course_name}:")
                logger.info(f"    📄 PDFs: {stats.total_docs}")
                logger.info(f"    📝 Chunks: {stats.new_chunks}")
                logger.info(f"    ⏱️ Tiempo: {stats.processing_time:.2f}s")
                if stats.errors:
                    logger.info(f"    ❌ Errores: {len(stats.errors)}")


# ---------- 8. CLI y Testing ------------------------------------------------
def test_pdf_processing():
    """Función de prueba para verificar el procesamiento"""
    settings = Settings()
    
    logger.info("🧪 Modo de prueba - Verificando configuración")
    logger.info(f"📂 Ruta base: {settings.pdf_base_path}")
    logger.info(f"🗄️ Colección: {settings.collection}")
    logger.info(f"🌐 Qdrant URL: {settings.qdrant_url}")
    
    # Verificar si la ruta existe
    if not settings.pdf_base_path.exists():
        logger.error(f"❌ La ruta {settings.pdf_base_path} no existe")
        return False
    
    # Escanear PDFs
    scanner = CourseScanner(settings.pdf_base_path)
    courses = scanner.scan_courses()
    
    if not courses:
        logger.warning("⚠️ No se encontraron PDFs para procesar")
        return False
    
    logger.info("✅ Configuración válida - Listo para procesar")
    return True


# ---------- 9. Función Principal --------------------------------------------
def main():
    """Función principal para ejecutar el procesamiento"""
    logger.info("🎯 Procesador de Material Educativo para Qdrant")
    logger.info("=" * 60)
    
    # Verificar dependencias
    if not PYMUPDF_AVAILABLE and not PYPDF2_AVAILABLE:
        logger.error("❌ No hay librerías de PDF disponibles. Instala PyMuPDF o PyPDF2")
        return False
    
    # Configuración
    settings = Settings()
    
    # Verificar configuración
    if not test_pdf_processing():
        logger.error("❌ Error en la configuración")
        return False
    
    # Crear manager y procesar
    try:
        manager = QdrantEducationalManager(settings)
        logger.info("🚀 Iniciando procesamiento completo...")
        
        results = manager.process_all_courses()
        
        if results:
            logger.info("🎉 Procesamiento completado exitosamente")
            return True
        else:
            logger.warning("⚠️ Procesamiento completado sin resultados")
            return False
            
    except Exception as e:
        logger.error(f"💥 Error crítico durante el procesamiento: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Proceso completado exitosamente")
    else:
        print("\n❌ Proceso terminado con errores") 