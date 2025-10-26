import os
import streamlit as st
import boto3
from botocore.exceptions import ClientError
import json
from bedrock_utils import query_knowledge_base, generate_response, valid_prompt


# Streamlit UI
st.title("Bedrock Chat Application")

# Sidebar for configurations
st.sidebar.header("Configuration")
model_id = st.sidebar.selectbox("Select LLM Model", ["anthropic.claude-3-haiku-20240307-v1:0", "anthropic.claude-3-5-sonnet-20240620-v1:0"])

default_kb_id = os.getenv("BEDROCK_KB_ID", "Z5SKDHV7SC")
kb_id = st.sidebar.text_input("Knowledge Base ID", default_kb_id)
temperature = st.sidebar.select_slider("Temperature", [i/10 for i in range(0,11)],1)
top_p = st.sidebar.select_slider("Top_P", [i/1000 for i in range(0,1001)], 1)
show_context = st.sidebar.checkbox("Show retrieved context", value=False)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if valid_prompt(prompt, model_id):
        # Query Knowledge Base
        kb_results = query_knowledge_base(prompt, kb_id)
        
        # Prepare context from Knowledge Base results
        context = "\n\n---\n\n".join([result['text'] for result in kb_results if result.get('text')])
        references = []
        for res in kb_results:
            source = res.get("source")
            if source:
                references.append(
                    f"{source} | score: {round(res['score'], 3) if res.get('score') else 'n/a'}"
                )
        
        # Generate response using LLM
        instruction = (
            "You are a heavy machinery assistant. Answer the user's question ONLY using the context. "
            "If the context does not contain the answer, reply with \"I don't have that information.\" "
            "Respond in the same language as the question."
        )
        full_prompt = f"{instruction}\n\nContext:\n{context or 'N/A'}\n\nQuestion: {prompt}\nAnswer:"
        response = generate_response(full_prompt, model_id, temperature, top_p)

        # Optionally surface retrieved references under the answer
        if references:
            response = f"{response}\n\n---\nRetrieved references:\n" + "\n".join(references)
        elif not context:
            response = f"I could not find relevant passages in Knowledge Base `{kb_id}`. Please verify the ID or ingest data first."
    else:
        response = "I'm unable to answer this, please try again"
    
    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

    if show_context and kb_results:
        with st.expander("Retrieved context"):
            for idx, res in enumerate(kb_results, start=1):
                st.markdown(f"**Result {idx}** | score: {res.get('score')} | source: {res.get('source')}")
                st.write(res.get('text', ''))
