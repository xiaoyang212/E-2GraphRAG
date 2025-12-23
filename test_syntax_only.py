"""
Syntax-only test for Dual-Index GraphRAG implementation
Tests that all Python files have valid syntax without importing dependencies
"""

import sys
import py_compile
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def test_file_syntax(filepath):
    """Test if a Python file has valid syntax"""
    try:
        py_compile.compile(filepath, doraise=True)
        return True
    except py_compile.PyCompileError as e:
        logger.error(f"Syntax error in {filepath}:")
        logger.error(str(e))
        return False


def main():
    logger.info("=" * 60)
    logger.info("DUAL-INDEX GRAPHRAG SYNTAX VALIDATION")
    logger.info("=" * 60 + "\n")
    
    files_to_test = [
        "dual_index_builder.py",
        "dual_retriever.py",
        "main_dual_index.py",
        "test_dual_index.py"
    ]
    
    results = {}
    for filepath in files_to_test:
        if not os.path.exists(filepath):
            logger.error(f"✗ File not found: {filepath}")
            results[filepath] = False
            continue
        
        logger.info(f"Testing: {filepath}")
        results[filepath] = test_file_syntax(filepath)
        if results[filepath]:
            logger.info(f"  ✓ Valid syntax\n")
        else:
            logger.error(f"  ✗ Syntax error\n")
    
    # Summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for filepath, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {filepath}")
    
    logger.info("-" * 60)
    logger.info(f"Total: {passed}/{total} files have valid syntax")
    logger.info("=" * 60)
    
    if passed == total:
        logger.info("\n✅ All files have valid Python syntax!")
        return 0
    else:
        logger.error(f"\n❌ {total - passed} file(s) have syntax errors.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
