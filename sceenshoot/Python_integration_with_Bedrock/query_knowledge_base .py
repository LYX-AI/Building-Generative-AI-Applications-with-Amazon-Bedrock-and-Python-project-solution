def query_knowledge_base(query, kb_id, top_k=3):
    try:
        response = bedrock_kb.retrieve(
            knowledgeBaseId=kb_id,
            retrievalQuery={
                'text': query
            },
            retrievalConfiguration={
                'vectorSearchConfiguration': {
                    'numberOfResults': top_k
                }
            }
        )
        formatted_results = []
        for item in response.get('retrievalResults', []):
            segments = []
            content = item.get('content', [])

            if isinstance(content, dict):
                candidate = content.get('text') or str(content)
                if candidate:
                    segments.append(candidate)
            elif isinstance(content, list):
                for segment in content:
                    if isinstance(segment, dict):
                        text = segment.get('text')
                    else:
                        text = str(segment)
                    if text:
                        segments.append(text)
            elif content:
                segments.append(str(content))

            source_uri = None
            location = item.get("location", {})
            if isinstance(location, dict):
                if location.get("type") == "S3":
                    source_uri = location.get("s3Location", {}).get("uri")
                else:
                    source_uri = location.get("type")
            metadata = item.get("metadata", {}) or {}
            if not source_uri and isinstance(metadata, dict):
                source_uri = metadata.get("source")

            formatted_results.append(
                {
                    "text": "\n".join(segments),
                    "score": item.get("score"),
                    "metadata": metadata,
                    "source": source_uri,
                    "location": location,
                    "raw": item,
                }
            )

        return formatted_results
    except ClientError as e:
        print(f"Error querying Knowledge Base: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error querying Knowledge Base: {e}")
        return []
