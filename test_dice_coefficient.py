#!/usr/bin/env python3
"""
Test for Dice coefficient edge weight calculation
"""

import sys
import numpy as np

def test_dice_coefficient():
    """Test that edge weights are calculated using Dice coefficient with chunks"""
    print("Testing Dice coefficient edge weight calculation...")
    
    # Mock data
    concept_vectors = {
        "concept_a": np.array([1.0, 0.0, 0.0]),
        "concept_b": np.array([0.9, 0.1, 0.0]),  # Similar to concept_a
    }
    
    # Concept A appears in sentences 0, 1, 2, 3 which map to chunks 0, 0, 1, 2 (3 unique chunks)
    # Concept B appears in sentences 0, 1, 4, 5, 6 which map to chunks 0, 0, 2, 3, 3 (3 unique chunks)
    I_c_to_s = {
        "concept_a": ["sent_0", "sent_1", "sent_2", "sent_3"],
        "concept_b": ["sent_0", "sent_1", "sent_4", "sent_5", "sent_6"],
    }
    
    I_s_to_c = {
        "sent_0": "leaf_0",  # Both concepts appear in chunk 0
        "sent_1": "leaf_0",  # Both concepts appear in chunk 0
        "sent_2": "leaf_1",  # Concept A in chunk 1
        "sent_3": "leaf_2",  # Concept A in chunk 2
        "sent_4": "leaf_2",  # Concept B in chunk 2
        "sent_5": "leaf_3",  # Concept B in chunk 3
        "sent_6": "leaf_3",  # Concept B in chunk 3
    }
    
    cooccurrence = {
        ("concept_a", "concept_b"): 2  # Co-occurs 2 times
    }
    
    # Import after setting up mock data
    from dual_index_graphrag import DualIndexBuilder
    from unittest.mock import MagicMock
    
    # Create a builder with mocked embedder
    builder = DualIndexBuilder(embedder_model="BAAI/bge-m3", device="cpu")
    builder.embedder = MagicMock()
    
    # Build graph
    G = builder.build_graph_with_vectors(
        concept_vectors=concept_vectors,
        I_c_to_s=I_c_to_s,
        I_s_to_c=I_s_to_c,
        cooccurrence=cooccurrence,
        theta_sem=0.3,
        theta_co=1
    )
    
    # Verify edge exists
    if not G.has_edge("concept_a", "concept_b"):
        print("✗ FAIL: Edge not created between concepts")
        return False
    
    # Get edge weight
    edge_weight = G["concept_a"]["concept_b"]["weight"]
    
    # Calculate expected Dice coefficient
    # r(w_i, w_j) = 2 * Co(w_i, w_j) / (|T_{w_i}| + |T_{w_j}|)
    # Concept A appears in chunks: {0, 1, 2} = 3 unique chunks
    # Concept B appears in chunks: {0, 2, 3} = 3 unique chunks
    # r = 2 * 2 / (3 + 3) = 4 / 6 ≈ 0.667
    expected_weight = (2 * 2) / (3 + 3)
    
    print(f"Edge weight: {edge_weight}")
    print(f"Expected Dice coefficient: {expected_weight}")
    
    # Check if weights match (with small tolerance for floating point)
    if abs(edge_weight - expected_weight) < 0.0001:
        print("✓ PASS: Edge weight correctly calculated using Dice coefficient")
        print(f"  Co-occurrence: 2")
        print(f"  |T_a|: 3 chunks (leaf_0, leaf_1, leaf_2)")
        print(f"  |T_b|: 3 chunks (leaf_0, leaf_2, leaf_3)")
        print(f"  Weight = 2 * 2 / (3 + 3) = {expected_weight:.4f}")
        return True
    else:
        print(f"✗ FAIL: Edge weight incorrect. Got {edge_weight}, expected {expected_weight}")
        return False


if __name__ == "__main__":
    try:
        success = test_dice_coefficient()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"✗ FAIL: Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
