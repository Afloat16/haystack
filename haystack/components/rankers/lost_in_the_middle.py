# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0


from haystack import Document, component
from haystack.utils.misc import _deduplicate_documents


@component
class LostInTheMiddleRanker:
    """
    A LostInTheMiddle Ranker.

    Ranks documents based on the 'lost in the middle' order so that the most relevant documents are either at the
    beginning or end, while the least relevant are in the middle.

    LostInTheMiddleRanker assumes that some prior component in the pipeline has already ranked documents by relevance
    and requires no query as input but only documents. It is typically used as the last component before building a
    prompt for an LLM to prepare the input context for the LLM.

    Lost in the Middle ranking lays out document contents into LLM context so that the most relevant contents are at
    the beginning or end of the input context, while the least relevant is in the middle of the context. See the
    paper ["Lost in the Middle: How Language Models Use Long Contexts"](https://arxiv.org/abs/2307.03172) for more
    details.

    Usage example:
    ```python
    from haystack.components.rankers import LostInTheMiddleRanker
    from haystack import Document

    ranker = LostInTheMiddleRanker()
    docs = [Document(content="Paris"), Document(content="Berlin"), Document(content="Madrid")]
    result = ranker.run(documents=docs)
    for doc in result["documents"]:
        print(doc.content)
    ```
    """

    def __init__(self, word_count_threshold: int | None = None, top_k: int | None = None) -> None:
        """
        Initialize the LostInTheMiddleRanker.

        :param word_count_threshold:
            The cumulative word count threshold used to select documents before applying the lost-in-the-middle
            reordering. Documents are considered in their original relevance order, after deduplication and `top_k`
            filtering. The document that causes the cumulative word count to reach or exceed the threshold is included
            in full, so the final word count may exceed the threshold. All subsequent documents are excluded.
            Words are counted using `str.split()`, not model tokens.
        :param top_k: The maximum number of documents to return.
        """
        if isinstance(word_count_threshold, int) and word_count_threshold <= 0:
            raise ValueError(
                f"Invalid value for word_count_threshold: {word_count_threshold}. word_count_threshold must be > 0."
            )
        if isinstance(top_k, int) and top_k <= 0:
            raise ValueError(f"top_k must be > 0, but got {top_k}")

        self.word_count_threshold = word_count_threshold
        self.top_k = top_k

    @component.output_types(documents=list[Document])
    def run(
        self, documents: list[Document], top_k: int | None = None, word_count_threshold: int | None = None
    ) -> dict[str, list[Document]]:
        """
        Reranks documents based on the "lost in the middle" order.

        Before ranking, documents are deduplicated by their id, retaining only the document with the highest score
        if a score is present.

        :param documents: List of Documents to reorder.
        :param top_k: The maximum number of documents to return.
        :param word_count_threshold:
            The cumulative word count threshold used to select documents before applying the lost-in-the-middle
            reordering. Documents are considered in their original relevance order, after deduplication and `top_k`
            filtering. The document that causes the cumulative word count to reach or exceed the threshold is included
            in full, so the final word count may exceed the threshold. All subsequent documents are excluded.
            Words are counted using `str.split()`, not model tokens.
        :returns:
            A dictionary with the following keys:
            - `documents`: Reranked list of Documents

        :raises ValueError:
            If any of the documents is not textual.
        """
        if isinstance(word_count_threshold, int) and word_count_threshold <= 0:
            raise ValueError(
                f"Invalid value for word_count_threshold: {word_count_threshold}. word_count_threshold must be > 0."
            )
        if isinstance(top_k, int) and top_k <= 0:
            raise ValueError(f"top_k must be > 0, but got {top_k}")

        if not documents:
            return {"documents": []}

        # Raise an error if any document is not textual
        if any(doc.content is None for doc in documents):
            raise ValueError("Some provided documents are not textual; LostInTheMiddleRanker can process only text.")

        top_k = top_k or self.top_k
        word_count_threshold = word_count_threshold or self.word_count_threshold

        deduplicated_documents = _deduplicate_documents(documents)
        documents_to_reorder = deduplicated_documents[:top_k] if top_k else deduplicated_documents

        # If there's only one document, return it as is
        if len(documents_to_reorder) == 1:
            return {"documents": documents_to_reorder}

        if word_count_threshold:
            word_count = 0
            for doc_idx, doc in enumerate(documents_to_reorder):
                if doc.content:
                    word_count += len(doc.content.split())
                    if word_count >= word_count_threshold:
                        # Include the complete document that reaches the threshold.
                        documents_to_reorder = documents_to_reorder[: doc_idx + 1]
                        break

        # Alternating relevance positions form the two ends of the context.
        # Build them directly instead of repeatedly shifting the middle of a list.
        ranked_docs = documents_to_reorder[::2]
        ranked_docs.extend(reversed(documents_to_reorder[1::2]))
        return {"documents": ranked_docs}
