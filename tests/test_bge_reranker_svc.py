import unittest
from app import reranker_bge_svc, config

class TestBGERerankerService(unittest.TestCase):
    def test_singleton_loading(self):
        """Verify that the BGE reranker model can be fetched as a singleton."""
        try:
            model = reranker_bge_svc.get_reranker()
            self.assertIsNotNone(model)
        except Exception as e:
            # If FlagEmbedding is not fully working or HuggingFace is offline, skip test
            self.skipTest(f"Skipping test due to model loading error: {e}")

    def test_rerank_basic(self):
        """Test BGE reranking on a simple query and set of documents."""
        try:
            _ = reranker_bge_svc.get_reranker()
        except Exception as e:
            self.skipTest(f"Skipping test due to model loading error: {e}")

        papers = [
            {
                "arxiv_id": "0001",
                "title": "Quantum Mechanics and Computing",
                "abstract": "An introduction to quantum physics and computing devices."
            },
            {
                "arxiv_id": "0002",
                "title": "Natural Language Processing with Transformers",
                "abstract": "We describe modern language modeling using neural network architectures."
            }
        ]

        # Query about language models should rank NLP paper first
        query = "neural network language models"
        reranked, elapsed = reranker_bge_svc.rerank(query, papers)

        self.assertEqual(len(reranked), 2)
        self.assertGreater(elapsed, 0)
        
        # The NLP paper (0002) should have a higher score than the quantum paper (0001)
        self.assertIn("bge_rerank_score", reranked[0])
        self.assertIn("bge_rerank_score", reranked[1])
        
        # The first paper in reranked should be 0002 because of language model match
        self.assertEqual(reranked[0]["arxiv_id"], "0002")

    def test_rerank_empty_and_fallback(self):
        """Test reranking under empty or invalid inputs."""
        papers = [{"arxiv_id": "0001", "title": "Test Title", "abstract": "Test Abstract"}]
        
        # Empty query
        res, elapsed = reranker_bge_svc.rerank("", papers)
        self.assertEqual(res, papers)
        self.assertEqual(elapsed, 0)

        # Empty papers
        res, elapsed = reranker_bge_svc.rerank("query", [])
        self.assertEqual(res, [])
        self.assertEqual(elapsed, 0)
