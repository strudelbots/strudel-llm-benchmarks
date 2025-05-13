# Use the Conversation API to send a text message to Mistral.

import boto3
from botocore.exceptions import ClientError

#session = boto3.Session(profile_name="bedrock_admin")
# Create a Bedrock Runtime client in the AWS Region you want to use.
#client = boto3.client("bedrock-runtime", region_name="us-east-1")
session = boto3.Session(profile_name="bedrock_admin")
client  = session.client('bedrock-runtime', region_name='us-east-1')

# Set the model ID, e.g., Mistral Large.
model_id = "mistral.mistral-7b-instruct-v0:2"
# Start a conversation with the user message.
user_message = "Describe the purpose of a 'hello world' program in one line."
conversation = [
    {
        "role": "user",
        "content": [{"text": user_message}],
    }
]

try:
    # Send the message to the model, using a basic inference configuration.
    response = client.converse(
        modelId=model_id,
        messages=conversation,
        inferenceConfig={"maxTokens": 512, "temperature": 0.5, "topP": 0.9},
    )

    # Extract and print the response text.
    response_text = response["output"]["message"]["content"][0]["text"]
    print(response_text)

except (ClientError, Exception) as e:
    print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
    exit(1)


