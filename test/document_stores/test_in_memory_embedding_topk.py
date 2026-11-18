# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Literal

import pytest

from haystack import Document, Pipeline
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore


@pytest.mark.parametrize("similarity", ["dot_product", "cosine"])
@pytest.mark.parametrize("scale_score", [False, True])
@pytest.mark.parametrize("top_k", [0, 1, 3, 10, 100, 120])
@pytest.mark.parametrize("filtered", [False, True])
def test_embedding_topk_preserves_scores_and_stable_ties(
    similarity: Literal["dot_product", "cosine"], scale_score: bool, top_k: int, filtered: bool
) -> None:
    # The query is the positive x axis. Repeated directions and magnitudes create ties,
    # including ties that straddle the requested cutoff. IDs deliberately do not sort by rank.
    vectors = [(3.0, 4.0), (1.0, 0.0), (0.0, 0.0), (-1.0, 0.0), (6.0, 8.0)]
    vectors += [(2.0, 0.0), (1.0, 0.0), (-2.0, 0.0), (0.0, 2.0), (10.0, 0.0)]
    vectors *= 10
    docs = [
        Document(
            id=f"doc-{len(vectors) - i}",
            content=str(i),
            embedding=list(vector),
            meta={"selected": i % 2 == 1},
            score=-111.0,
        )
        for i, vector in enumerate(vectors)
    ]
    missing_embedding = Document(id="no-embedding", content="no vector", meta={"selected": True})
    store = InMemoryDocumentStore(embedding_similarity_function=similarity, shared=False)
    try:
        store.write_documents([docs[0], missing_embedding, *docs[1:]])
        filters = {"field": "meta.selected", "operator": "==", "value": True} if filtered else None
        actual = store.embedding_retrieval(
            query_embedding=[1.0, 0.0], filters=filters, top_k=top_k, scale_score=scale_score
        )

        # Independently calculate the scalar projection on the query direction.
        expected = []
        for i, (x, y) in enumerate(vectors):
            if filtered and i % 2 != 1:
                continue
            score = x if similarity == "dot_product" else x / (math.hypot(x, y) or 1.0)
            if scale_score:
                score = 1.0 / (1.0 + math.exp(-score / 100.0)) if similarity == "dot_product" else (score + 1.0) / 2.0
            expected.append((docs[i].id, score))
        expected.sort(key=lambda item: item[1], reverse=True)
        expected = expected[:top_k]
        assert [doc.id for doc in actual] == [doc_id for doc_id, _ in expected]
        assert [doc.score for doc in actual] == pytest.approx([score for _, score in expected])
        assert all(doc.embedding is None for doc in actual)
        assert all(
            doc.score == -111.0 and doc.embedding == list(vector) for doc, vector in zip(docs, vectors, strict=True)
        )
    finally:
        store.shutdown()


@pytest.mark.parametrize(
    "scores",
    [
        [math.inf, 2.0, math.inf, -math.inf, 0.0, -0.0],
        [math.nan, 2.0, 1.0, math.inf, -math.inf, 0.0],
        [2.0, math.nan, 1.0, math.inf, -math.inf, 0.0],
        [2.0, 1.0, math.inf, -math.inf, 0.0, math.nan],
        [2.0, math.nan, 1.0, math.nan, -math.inf, math.inf],
    ],
)
@pytest.mark.parametrize("top_k", [0, 1, 3, 6, 120, 150])
def test_embedding_topk_preserves_non_finite_order(scores: list[float], top_k: int) -> None:
    scores = scores * 20
    store = InMemoryDocumentStore(shared=False)
    try:
        store.write_documents([Document(id=str(i), embedding=[score]) for i, score in enumerate(scores)])
        actual = store.embedding_retrieval(query_embedding=[1.0], top_k=top_k)
        # Preserve the existing full-sort result even when NaN does not define a total order.
        expected_indices = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:top_k]
        assert [doc.id for doc in actual] == [str(i) for i in expected_indices]
        assert [doc.score for doc in actual] == pytest.approx([scores[i] for i in expected_indices], nan_ok=True)
    finally:
        store.shutdown()


def test_embedding_topk_pipeline_preserves_saturated_score_ties() -> None:
    store = InMemoryDocumentStore(shared=False)
    try:
        # All positive scores saturate to 1.0; ranking must use the scaled score,
        # preserving insertion order rather than preferring the largest raw dot product.
        store.write_documents(
            [
                Document(id=doc_id, embedding=[score])
                for doc_id, score in [("first", 1e5), ("second", 2e5), ("third", 3e5)]
            ]
            + [Document(id=f"other-{i}", embedding=[0.0]) for i in range(30)]
        )
        pipeline = Pipeline()
        pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store=store, top_k=2, scale_score=True))
        documents = pipeline.run({"retriever": {"query_embedding": [1.0]}})["retriever"]["documents"]
        assert [(doc.id, doc.score) for doc in documents] == [("first", 1.0), ("second", 1.0)]
    finally:
        store.shutdown()
