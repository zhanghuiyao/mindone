"""
Basic syntax and structure test for glmasr model.
This test verifies that the model files are properly structured without
requiring full mindone.transformers import due to compatibility issues.
"""
import ast
import os


def test_modeling_file_structure():
    """Test that modeling_glmasr.py has the expected structure."""
    file_path = os.path.join(
        os.path.dirname(__file__),
        "../../../../mindone/transformers/models/glmasr/modeling_glmasr.py"
    )
    
    with open(file_path, 'r') as f:
        tree = ast.parse(f.read())
    
    # Get all class names
    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    
    # Expected classes
    expected_classes = [
        'GlmAsrRotaryEmbedding',
        'GlmAsrAttention',
        'GlmAsrMLP',
        'GlmAsrEncoderLayer',
        'GlmAsrPreTrainedModel',
        'GlmAsrEncoder',
        'GlmAsrMultiModalProjector',
        'GlmAsrForConditionalGeneration',
    ]
    
    for expected_class in expected_classes:
        assert expected_class in classes, f"Missing class: {expected_class}"
    
    print(f"✓ All expected classes found: {expected_classes}")
    
    # Get all function names
    functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    
    expected_functions = [
        'rotate_half',
        'repeat_kv',
        'eager_attention_forward',
        'apply_rotary_pos_emb',
    ]
    
    for expected_func in expected_functions:
        assert expected_func in functions, f"Missing function: {expected_func}"
    
    print(f"✓ All expected functions found: {expected_functions}")


def test_configuration_file_structure():
    """Test that configuration_glmasr.py has the expected structure."""
    file_path = os.path.join(
        os.path.dirname(__file__),
        "../../../../mindone/transformers/models/glmasr/configuration_glmasr.py"
    )
    
    with open(file_path, 'r') as f:
        tree = ast.parse(f.read())
    
    # Get all class names
    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    
    # Expected classes
    expected_classes = [
        'GlmAsrEncoderConfig',
        'GlmAsrConfig',
    ]
    
    for expected_class in expected_classes:
        assert expected_class in classes, f"Missing class: {expected_class}"
    
    print(f"✓ Configuration classes found: {expected_classes}")


def test_processing_file_structure():
    """Test that processing_glmasr.py has the expected structure."""
    file_path = os.path.join(
        os.path.dirname(__file__),
        "../../../../mindone/transformers/models/glmasr/processing_glmasr.py"
    )
    
    with open(file_path, 'r') as f:
        tree = ast.parse(f.read())
    
    # Get all class names
    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    
    # Expected classes
    expected_classes = [
        'GlmAsrProcessorKwargs',
        'GlmAsrProcessor',
    ]
    
    for expected_class in expected_classes:
        assert expected_class in classes, f"Missing class: {expected_class}"
    
    print(f"✓ Processor classes found: {expected_classes}")


def test_mindspore_usage():
    """Test that files use mindspore instead of torch."""
    file_path = os.path.join(
        os.path.dirname(__file__),
        "../../../../mindone/transformers/models/glmasr/modeling_glmasr.py"
    )
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check for mindspore usage
    assert 'import mindspore' in content, "Missing mindspore import"
    assert 'mindspore.nn.Cell' in content, "Missing mindspore.nn.Cell usage"
    assert 'def construct(' in content, "Missing construct method"
    
    # Make sure torch is not imported (except in comments)
    lines = [line for line in content.split('\n') if not line.strip().startswith('#')]
    code_content = '\n'.join(lines)
    assert 'import torch' not in code_content, "Should not import torch in model file"
    
    print("✓ File correctly uses mindspore instead of torch")


if __name__ == "__main__":
    test_modeling_file_structure()
    test_configuration_file_structure()
    test_processing_file_structure()
    test_mindspore_usage()
    print("\n✓✓✓ All basic structure tests passed! ✓✓✓")
