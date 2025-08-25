"""
AST parsing utility functions
"""

import ast
from typing import List, Dict, Any

def parse_script(script_content: str) -> ast.AST:
    """
    Safely parse Python script into AST
    """
    try:
        return ast.parse(script_content)
    except SyntaxError as e:
        raise ValueError(f"Script has syntax errors: {e}")

def extract_functions(tree: ast.AST) -> List[str]:
    """
    Extract all function names from AST
    """
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            functions.append(node.name)
    return functions

def find_class_definitions(tree: ast.AST) -> List[Dict[str, Any]]:
    """
    Find all class definitions in the AST
    """
    classes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_info = {
                'name': node.name,
                'methods': [],
                'base_classes': []
            }
            
            # Extract base classes
            for base in node.bases:
                if isinstance(base, ast.Name):
                    class_info['base_classes'].append(base.id)
            
            # Extract methods
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    class_info['methods'].append(item.name)
            
            classes.append(class_info)
    
    return classes