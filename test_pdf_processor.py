#!/usr/bin/env python3
"""
Script de prueba para el procesador de PDFs educativos
Verifica la configuración y realiza pruebas básicas
"""

import sys
import os
from pathlib import Path

# Agregar el directorio padre al path para importar el módulo
sys.path.append(str(Path(__file__).parent))

from pdf_educational_processor import (
    Settings, 
    CourseScanner, 
    PDFProcessor, 
    QdrantEducationalManager,
    test_pdf_processing,
    logger
)

def test_dependencies():
    """Prueba las dependencias necesarias"""
    logger.info("🔍 Verificando dependencias...")
    
    missing_deps = []
    
    try:
        import fitz
        logger.info("✅ PyMuPDF disponible")
    except ImportError:
        missing_deps.append("PyMuPDF")
        logger.warning("⚠️ PyMuPDF no disponible")
    
    try:
        import PyPDF2
        logger.info("✅ PyPDF2 disponible")
    except ImportError:
        missing_deps.append("PyPDF2")
        logger.warning("⚠️ PyPDF2 no disponible")
    
    try:
        from qdrant_client import QdrantClient
        logger.info("✅ Qdrant client disponible")
    except ImportError:
        missing_deps.append("qdrant-client")
        logger.error("❌ Qdrant client no disponible")
    
    try:
        from langchain_aws import BedrockEmbeddings
        logger.info("✅ LangChain AWS disponible")
    except ImportError:
        missing_deps.append("langchain-aws")
        logger.error("❌ LangChain AWS no disponible")
    
    if missing_deps:
        logger.error(f"❌ Dependencias faltantes: {', '.join(missing_deps)}")
        logger.info("💡 Instalar con: pip install -r requirements_pdf.txt")
        return False
    
    logger.info("✅ Todas las dependencias están disponibles")
    return True

def test_configuration():
    """Prueba la configuración"""
    logger.info("⚙️ Verificando configuración...")
    
    try:
        settings = Settings()
        logger.info(f"📂 Ruta base PDFs: {settings.pdf_base_path}")
        logger.info(f"🗄️ Colección Qdrant: {settings.collection}")
        logger.info(f"🌐 URL Qdrant: {settings.qdrant_url}")
        logger.info(f"🔢 Tamaño de chunk: {settings.chunk_size}")
        logger.info(f"📦 Tamaño de batch: {settings.batch_size}")
        
        if not settings.pdf_base_path.exists():
            settings.pdf_base_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 Creada carpeta: {settings.pdf_base_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en configuración: {e}")
        return False

def test_course_scanner():
    """Prueba el escáner de cursos"""
    logger.info("🔍 Probando escáner de cursos...")
    
    try:
        settings = Settings()
        scanner = CourseScanner(settings.pdf_base_path)
        courses = scanner.scan_courses()
        
        if courses:
            logger.info(f"📚 Cursos encontrados: {len(courses)}")
            for course, files in courses.items():
                logger.info(f"  - {course}: {len(files)} archivos")
        else:
            logger.info("📚 No se encontraron cursos (esto es normal si no hay PDFs)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en escáner: {e}")
        return False

def test_qdrant_connection():
    """Prueba la conexión a Qdrant"""
    logger.info("🔗 Probando conexión a Qdrant...")
    
    try:
        settings = Settings()
        from qdrant_client import QdrantClient
        
        client = QdrantClient(url=settings.qdrant_url)
        
        # Intentar obtener info del cluster
        cluster_info = client.get_cluster_info()
        logger.info(f"✅ Conectado a Qdrant: {cluster_info}")
        
        # Listar colecciones
        collections = client.get_collections()
        logger.info(f"📋 Colecciones existentes: {len(collections.collections)}")
        
        return True
        
    except Exception as e:
        logger.warning(f"⚠️ No se pudo conectar a Qdrant: {e}")
        logger.info("💡 Asegúrate de que Qdrant esté ejecutándose en localhost:6333")
        return False

def create_sample_pdf_structure():
    """Crea una estructura de ejemplo para testing"""
    logger.info("📁 Creando estructura de ejemplo...")
    
    settings = Settings()
    base_path = settings.pdf_base_path
    
    # Crear estructura de carpetas de ejemplo
    sample_courses = [
        "matematicas/primaria",
        "matematicas/secundaria", 
        "historia/civica",
        "ciencias/biologia",
        "lengua/literatura"
    ]
    
    for course in sample_courses:
        course_path = base_path / course
        course_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"📂 Creada: {course_path}")
        
        # Crear archivo README en cada carpeta
        readme_file = course_path / "README.txt"
        readme_file.write_text(
            f"Carpeta para el curso: {course}\n"
            f"Coloca aquí los archivos PDF del curso.\n"
            f"Creado automáticamente por el test.\n"
        )
    
    logger.info("✅ Estructura de ejemplo creada")
    logger.info(f"💡 Coloca tus PDFs en las carpetas bajo: {base_path}")

def run_full_test():
    """Ejecuta todas las pruebas"""
    logger.info("🚀 Iniciando pruebas completas del procesador de PDFs educativos")
    logger.info("=" * 70)
    
    tests = [
        ("Dependencias", test_dependencies),
        ("Configuración", test_configuration),
        ("Escáner de cursos", test_course_scanner),
        ("Conexión Qdrant", test_qdrant_connection)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Ejecutando: {test_name}")
        logger.info("-" * 40)
        
        try:
            result = test_func()
            results[test_name] = result
            
            if result:
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.warning(f"⚠️ {test_name}: FAILED")
                
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
            results[test_name] = False
    
    # Resumen final
    logger.info("\n" + "=" * 70)
    logger.info("📋 RESUMEN DE PRUEBAS")
    logger.info("=" * 70)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info("-" * 40)
    logger.info(f"Total: {passed}/{total} pruebas pasaron")
    
    if passed == total:
        logger.info("🎉 ¡Todas las pruebas pasaron! El sistema está listo.")
        
        # Crear estructura de ejemplo si no existe
        settings = Settings()
        if not any(settings.pdf_base_path.rglob("*.pdf")):
            create_sample_pdf_structure()
            
        return True
    else:
        logger.warning("⚠️ Algunas pruebas fallaron. Revisa la configuración.")
        return False

def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pruebas para el procesador de PDFs educativos")
    parser.add_argument("--deps", action="store_true", help="Solo probar dependencias")
    parser.add_argument("--config", action="store_true", help="Solo probar configuración")
    parser.add_argument("--scan", action="store_true", help="Solo probar escáner")
    parser.add_argument("--qdrant", action="store_true", help="Solo probar Qdrant")
    parser.add_argument("--create-structure", action="store_true", help="Crear estructura de ejemplo")
    
    args = parser.parse_args()
    
    if args.deps:
        return test_dependencies()
    elif args.config:
        return test_configuration()
    elif args.scan:
        return test_course_scanner()
    elif args.qdrant:
        return test_qdrant_connection()
    elif args.create_structure:
        create_sample_pdf_structure()
        return True
    else:
        return run_full_test()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 