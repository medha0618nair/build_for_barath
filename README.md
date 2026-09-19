# Cross-jurisdiction crime linkage

Links property-crime cases across state borders by behavioural similarity.
Surfaces a ranked shortlist for a human analyst. Never identifies a person.

Full write-up (story, architecture diagram, honest limits, cost breakdown)
is a Phase 7 deliverable and isn't written yet. This section covers what
Phase 5 (`infra/`, `handlers/`) needs: how to deploy it.

## Deploy

Prerequisites: AWS CLI v2, AWS SAM CLI, and credentials for an account with
permission to create the stack's resources (CloudFormation, S3, DynamoDB,
Lambda, Step Functions, EventBridge, API Gateway, Cognito, Verified
Permissions, KMS, IAM roles).

### One-time: Bedrock model access

Bedrock model access is opt-in per model, per region, and is **not
instant** — request it before you need it.

1. Console: **Amazon Bedrock → Model access** (in **ap-south-1**, Mumbai).
2. Click **Manage model access** (or **Modify model access**).
3. Check the boxes for the Claude Haiku model (used by
   `handlers/bedrock_extract.py`) and the Cohere Embed Multilingual model
   (used by `handlers/cohere_embed.py`).
4. Submit. Anthropic and Cohere's models are typically granted quickly, but
   this is a manual step outside CloudFormation's control — the pipeline's
   Enrich and BuildIndex steps will fail with an access-denied error from
   Bedrock until it's approved.
5. Verify the exact model IDs before deploying for real use — **never
   guess the versioned form** (CLAUDE.md):

   ```bash
   aws bedrock list-foundation-models --region ap-south-1 \
     --query "modelSummaries[?contains(modelId, 'claude') || contains(modelId, 'cohere')].modelId"
   aws bedrock list-inference-profiles --region ap-south-1
   ```

   `infra/template.yaml`'s `BedrockHaikuModelId` and `CohereEmbedModelId`
   parameters default to the **unverified short forms** from CLAUDE.md
   (`anthropic.claude-haiku-4-5`, `cohere.embed-multilingual-v3`). Pass the
   exact strings the commands above return with `--parameter-overrides` at
   deploy time if they differ.

### Build and deploy

```bash
cd infra
sam build
sam validate --lint
sam deploy --guided --stack-name crime-linkage-dev \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides BedrockHaikuModelId=<verified id> CohereEmbedModelId=<verified id>
```

`--guided` walks through region (use `ap-south-1`), confirms changesets,
and saves the answers to `infra/samconfig.toml` for future
non-interactive `sam deploy`.

### Seed data (so the API has something to answer with immediately)

The Step Functions pipeline (Normalise → Enrich → Index → Link) populates
`cases`/`links` from feed files landing in S3, but a full run over 44,533
cases is slow and costs real Bedrock/DynamoDB usage. For a fast working
demo, seed directly from the already-computed local artifacts instead:

```bash
BUCKET=$(aws cloudformation describe-stacks --stack-name crime-linkage-dev \
  --query "Stacks[0].Outputs[?OutputKey=='DataBucketName'].OutputValue" --output text)

python -m scripts.upload_config --bucket "$BUCKET"
python -m scripts.upload_model --bucket "$BUCKET"     # frequencies, priors, weights.json, seed vector index
python -m scripts.seed_dynamodb \
  --cases-table crime-linkage-dev-cases \
  --links-table crime-linkage-dev-links
```

To run the real pipeline instead (or in addition), upload a feed CSV under
`raw/` in the data bucket — EventBridge starts the Step Functions execution
automatically — or start it directly:

```bash
aws stepfunctions start-execution \
  --state-machine-arn "$(aws cloudformation describe-stacks --stack-name crime-linkage-dev \
    --query "Stacks[0].Outputs[?OutputKey=='PipelineStateMachineArn'].OutputValue" --output text)" \
  --input '{"feed_files": [{"bucket": "'"$BUCKET"'", "key": "raw/MH.csv", "state_code": "MH"}]}'
```

### Verify

```bash
API_URL=$(aws cloudformation describe-stacks --stack-name crime-linkage-dev \
  --query "Stacks[0].Outputs[?OutputKey=='ApiUrl'].OutputValue" --output text)

# Cognito JWT auth is required (infra/template.yaml's HttpApi authorizer) —
# create a user and get a token first:
aws cognito-idp sign-up --client-id <UserPoolClientId> --username demo@example.gov.in --password '<StrongPassw0rd!>'
aws cognito-idp admin-confirm-sign-up --user-pool-id <UserPoolId> --username demo@example.gov.in
TOKEN=$(aws cognito-idp initiate-auth --client-id <UserPoolClientId> --auth-flow USER_PASSWORD_AUTH \
  --auth-parameters USERNAME=demo@example.gov.in,PASSWORD='<StrongPassw0rd!>' \
  --query "AuthenticationResult.IdToken" --output text)

curl -H "Authorization: Bearer $TOKEN" "$API_URL/v1/cases/<a seeded case_id>"
```

## Local development (no AWS required)

Everything runs end to end locally first — AWS is Phase 5, not earlier.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m linkage.config                 # check config/
python -m linkage.normalise --check      # Phase 1 gate
python -m linkage.score --worked-example # Phase 2 gate
python -m linkage.train && python -m linkage.evaluate  # Phase 3
python -m linkage.link_batch             # Phase 4
python -m api.serve_fixtures             # local fixture server
pytest tests/
```
