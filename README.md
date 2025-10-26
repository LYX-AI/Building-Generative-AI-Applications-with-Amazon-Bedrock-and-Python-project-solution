cd13926-Building-Generative-AI-Applications-with-Amazon-Bedrock-and-Python-project-solution is the hole project code

screenshoot saved the screenshoot 

temperature_top_p_explanation.docx is the answer of explaining the effects of temperature and top_p


# Building Generative AI Applications with Amazon Bedrock and Python

This repository delivers an end-to-end sample for building a retrieval-augmented chat assistant on top of Amazon Bedrock. Terraform stacks provision the networking, Aurora PostgreSQL Serverless v2 cluster, S3 document bucket, IAM roles, and the Bedrock Knowledge Base. A Streamlit app (`app.py`) validates prompts, retrieves context from the knowledge base, and calls Anthropic Claude models hosted on Bedrock to answer user questions about heavy machinery.

## Contents

- [Architecture Overview](#architecture-overview)
- [Repository Layout](#repository-layout)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
  - [1. Provision network, database, and S3 (stack1)](#1-provision-network-database-and-s3-stack1)
  - [2. Prepare the Aurora database](#2-prepare-the-aurora-database)
  - [3. Provision the Bedrock Knowledge Base (stack2)](#3-provision-the-bedrock-knowledge-base-stack2)
  - [4. Ingest documents into S3](#4-ingest-documents-into-s3)
  - [5. Run the Streamlit chat application](#5-run-the-streamlit-chat-application)
- [Configuration Reference](#configuration-reference)
- [Operations and Maintenance](#operations-and-maintenance)
- [Troubleshooting](#troubleshooting)

## Architecture Overview

1. **Terraform stack1** creates foundational infrastructure in `us-west-2`: a VPC with public/private subnets, an Aurora PostgreSQL Serverless v2 cluster (with Data API enabled), and an S3 bucket for reference documents used as knowledge base sources.
2. **Terraform stack2** connects Bedrock to those resources by creating a Knowledge Base backed by Aurora vector storage plus an S3 data source. IAM roles and policies grant Bedrock permission to query Aurora via the Data API and access the S3 bucket.
3. **Streamlit application** provides a lightweight chat UI. Requests are checked by `bedrock_utils.valid_prompt` to ensure the user stays within the supported heavy-machinery domain before `query_knowledge_base` retrieves context and `generate_response` calls Anthropic Claude models through the Bedrock Runtime API.

## Repository Layout

```
.
├── app.py                     # Streamlit chat interface
├── bedrock_utils.py           # Bedrock helper functions (prompt guard, KB retrieval, text generation)
├── modules/
│   ├── bedrock_kb/            # Terraform module for the Bedrock Knowledge Base, IAM, and data source
│   └── database/              # Terraform module that provisions Aurora Serverless and Secrets Manager secret
├── requirements.txt           # Python dependencies
├── scripts/
│   ├── aurora_sql.sql         # SQL statements to enable pgvector and create KB tables/roles
│   ├── upload_s3.py           # Utility to upload source PDFs to the S3 bucket
│   └── spec-sheets/           # Sample spec sheets bundled with the repo
├── stack1/                    # Terraform stack for VPC, Aurora, and S3
└── stack2/                    # Terraform stack for the Bedrock Knowledge Base
```

Terraform state files (`terraform.tfstate` and backups) live inside each stack directory. Switch to remote state (e.g., S3 + DynamoDB) before collaborating with a team.

## Prerequisites

- AWS account with permissions for VPC, RDS, IAM, S3, Secrets Manager, and Bedrock.
- AWS CLI configured locally (`aws configure`) or relevant environment variables exported.
- Terraform version 1.5 or higher.
- Python 3.10+ and `pip`.
- Amazon Bedrock access in `us-west-2`, including Anthropic Claude models and the Knowledge Base feature.
- (Optional) `virtualenv`/`pyenv` for Python environment isolation.

## Getting Started

### 1. Provision network, database, and S3 (stack1)

1. Export or configure AWS credentials (for example `AWS_PROFILE`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`).
2. From the project root, change into `stack1/` and initialize Terraform:
   ```bash
   terraform init
   ```
3. Review `stack1/main.tf` and adjust CIDR blocks, Aurora sizing, or tags if needed.
4. Apply the stack:
   ```bash
   terraform apply
   ```
5. Capture the outputs (`terraform output`). You will need at least:
   - `aurora_arn`
   - `aurora_endpoint`
   - `rds_secret_arn`
   - `s3_bucket_name`
   - `db_endpoint` (optional for manual SQL access)

> If you already maintain a VPC or Aurora cluster, you can substitute your existing resources by editing the module inputs.

### 2. Prepare the Aurora database

1. Connect to the Aurora cluster using the RDS Query Editor or any SQL client that supports the Data API.
2. Execute the statements in `scripts/aurora_sql.sql`. They:
   - Enable the `vector` extension.
   - Create a `bedrock_integration` schema and `bedrock_user` role.
   - Create the `bedrock_integration.bedrock_kb` table with vector, text, and JSON metadata columns.
   - Add an HNSW index for efficient similarity search.
3. Confirm that the Secrets Manager secret created by `stack1` stores the database credentials for Bedrock to assume later.

### 3. Provision the Bedrock Knowledge Base (stack2)

1. Update `stack2/main.tf` with values from `stack1` outputs:
   - `aurora_arn`, `aurora_endpoint`, and `aurora_secret_arn`
   - `aurora_table_name` (`bedrock_integration.bedrock_kb`) and column names
   - `s3_bucket_arn` (e.g., `arn:aws:s3:::bedrock-kb-<account-id>`)
   - Optionally override `knowledge_base_name`
2. From `stack2/`, run:
   ```bash
   terraform init
   terraform apply
   ```
3. Record the outputs:
   - `bedrock_knowledge_base_id`
   - `bedrock_knowledge_base_arn`

These identifiers are required by the Streamlit application and when triggering data sync jobs.

### 4. Ingest documents into S3

1. Place your PDF or text assets inside `scripts/spec-sheets/` or provide another folder path when calling the upload script.
2. Edit `scripts/upload_s3.py` and set:
   - `bucket_name` to the bucket provisioned in stack1.
   - `prefix` if you want to maintain a folder prefix in S3 (defaults to `spec-sheets`).
3. Run the script from the project root (with AWS credentials available):
   ```bash
   python scripts/upload_s3.py
   ```
4. Navigate to the Bedrock console → Knowledge Bases → Data sources → **Sync** to generate fresh embeddings.

### 5. Run the Streamlit chat application

1. (Recommended) Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   source .venv/bin/activate  # macOS/Linux
   ```
2. Install dependencies from the project root:
   ```bash
   pip install -r requirements.txt
   ```
3. Export the required environment variables:
   ```bash
   set AWS_REGION=us-west-2
   set BEDROCK_KB_ID=<knowledge_base_id_from_stack2>
   ```
   The app defaults to `Z5SKDHV7SC` if `BEDROCK_KB_ID` is absent—override it with your deployment’s ID.
4. Launch the UI:
   ```bash
   streamlit run app.py
   ```
5. In the Streamlit sidebar, pick a Bedrock model (`anthropic.claude-3-haiku-20240307-v1:0` or `anthropic.claude-3-5-sonnet-20240620-v1:0`), adjust temperature and top-p, and start chatting. Enable **Show retrieved context** to inspect the passages returned by the knowledge base.

The app only responds to heavy-machinery questions. `bedrock_utils.valid_prompt` calls Bedrock to classify the user request; off-topic, policy-violating, or meta questions are declined gracefully.

## Configuration Reference

- **Region**: All Terraform providers and boto3 clients target `us-west-2`. Change the region in `stack1`, `stack2`, and `bedrock_utils.py` if you deploy elsewhere.
- **Environment variables**:
  - `BEDROCK_KB_ID`: Knowledge Base identifier for the chat app (required).
  - `AWS_PROFILE`, `AWS_REGION`: honored by boto3 if set.
- **Terraform variables**:
  - Customize Aurora sizing (`min_capacity`, `max_capacity`) and VPC CIDRs through `modules/database` and `stack1/main.tf`.
  - Edit IAM policies in `modules/bedrock_kb/main.tf` if you need stricter permissions.
- **Prompt guard**: Adjust the classification prompt inside `bedrock_utils.valid_prompt` to enforce different content categories.
- **Model list**: Extend the `model_id` select box in `app.py` with other Bedrock model IDs you are authorized to use.

## Operations and Maintenance

- **Data refresh**: Trigger a data source sync after uploading or modifying documents so embeddings stay current.
- **Secrets rotation**: Rotate the Aurora password via Secrets Manager or by reapplying the Terraform module (generates a new random password).
- **State management**: Adopt remote backend storage for Terraform state before collaborating in a team or CI/CD environment.
- **Logging and monitoring**: Enable CloudWatch logs for Aurora, Bedrock invocation logs, and Streamlit server logs if you require observability in production.

## Troubleshooting

- **Terraform apply fails**: Verify AWS credentials and permissions. Ensure the required Bedrock service quotas are available in the target region.
- **Database connectivity issues**: Confirm that the security group ingress (`allowed_cidr_blocks` in `modules/database/variables.tf`) permits your IP and that the Data API is enabled.
- **Knowledge Base returns no results**: Check that the data source sync succeeded, the Aurora table contains rows, and the Knowledge Base ID in the app matches Terraform outputs.
- **Streamlit app errors**: Make sure the virtual environment contains `boto3` and `streamlit`, and that the AWS identity running the app has Bedrock `InvokeModel` and Knowledge Base permissions.

For production workloads consider adding CI/CD automation, least-privilege IAM policies, centralized logging, and automated data ingestion pipelines.


