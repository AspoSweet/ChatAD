"""
LLM API utilities for TSEvol.

The evolution pipeline is provider-agnostic. Configure the backend LLM
through environment variables before running:

  Azure OpenAI:
    export LLM_PROVIDER=azure
    export AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com
    export AZURE_OPENAI_API_KEY=<your-key>
    export AZURE_OPENAI_API_VERSION=2025-03-01-preview
    export LLM_DEPLOYMENT=<your-deployment-name>

  OpenAI (or any OpenAI-compatible endpoint):
    export LLM_PROVIDER=openai
    export OPENAI_API_KEY=<your-key>
    export OPENAI_BASE_URL=<optional-custom-base-url>
    export LLM_DEPLOYMENT=<model-name>

The experiments in the paper use a reasoning-capable teacher model served
through the Responses API (referred to as GPT-5 in the paper).
"""
import os
import time

import openai
from openai import AzureOpenAI, OpenAI

API_MAX_RETRY = 3
API_ERROR_OUTPUT = "$ERROR$"
ANSWER_ERROR_OUTPUT = "$ANSWER_ERROR$"

DEFAULT_DEPLOYMENT = os.getenv("LLM_DEPLOYMENT", "gpt-5")
DEFAULT_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-03-01-preview")


def _build_client(api_version=None):
    """Create an OpenAI-compatible client from environment variables."""
    provider = os.getenv("LLM_PROVIDER", "azure").lower()
    if provider == "azure":
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not endpoint or not api_key:
            raise EnvironmentError(
                "Please set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY "
                "(or set LLM_PROVIDER=openai and OPENAI_API_KEY)."
            )
        return AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version or DEFAULT_API_VERSION,
        )
    # Standard OpenAI / OpenAI-compatible server
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
    )


class Model:
    """A thin wrapper around a reasoning-capable chat model."""

    def __init__(self, deployment_name=DEFAULT_DEPLOYMENT, instance=None,
                 endpoint=None, api_version=None):
        """
        :param deployment_name: deployment/model name of the backend LLM
        :param instance:        kept for backward compatibility (unused)
        :param endpoint:        kept for backward compatibility; the endpoint
                                is read from environment variables instead
        :param api_version:     Azure OpenAI API version (optional)
        """
        self.deployment_name = deployment_name
        self.instance = instance
        self.endpoint = endpoint
        self.api_version = api_version or DEFAULT_API_VERSION
        self.client = _build_client(self.api_version)

    def parse_response_GPT5(self, response):
        """Parse a Responses-API result from a reasoning model.

        :param response: the raw API response
        :return: (list of chain-of-thought summary steps, final answer,
                  [input_tokens, output_tokens])
        """
        cot_list = [data.text for data in response.output[0].summary]
        answer = response.output[1].content[0].text
        tokens_used = [response.usage.input_tokens, response.usage.output_tokens]
        return cot_list, answer, tokens_used

    def get_parsed_response_from_model(self, content):
        """Query the model with medium reasoning effort (used by agents)."""
        cotList, answer, token_used = ANSWER_ERROR_OUTPUT, ANSWER_ERROR_OUTPUT, 0
        for retry_i in range(API_MAX_RETRY):
            try:
                response = self.client.responses.create(
                    model=self.deployment_name,
                    store=True,
                    reasoning={"effort": "medium", "summary": "auto"},
                    text={"verbosity": "medium"},
                    input=[{"role": "user", "content": content}],
                )
                cotList, answer, token_used = self.parse_response_GPT5(response)
                return {'cotList': cotList, 'answer': answer, 'token_used': token_used}
            except openai.OpenAIError as e:  # retry up to API_MAX_RETRY times
                print(type(e), e)
            time.sleep(60)
        return {'cotList': cotList, 'answer': answer, 'token_used': token_used}


def get_parsed_response(content, Model):
    """Query the model with low reasoning effort (lightweight checks).

    :param content: input text (i.e., the question)
    :param Model:   a wrapped ``Model`` instance
    :return: (chain-of-thought list, final answer, tokens used)
    """
    cotList, answer, token_used = ANSWER_ERROR_OUTPUT, ANSWER_ERROR_OUTPUT, 0
    for retry_i in range(API_MAX_RETRY):
        try:
            response = Model.client.responses.create(
                model=Model.deployment_name,
                store=True,
                reasoning={"effort": "low", "summary": "low"},
                text={"verbosity": "low"},
                input=[{"role": "user", "content": content}],
            )
            cotList, answer, token_used = Model.parse_response_GPT5(response)
            break
        except openai.OpenAIError as e:  # retry up to API_MAX_RETRY times
            print(f"""{Model.deployment_name} encountered error""")
            print(type(e), e)
        time.sleep(121)
    return cotList, answer, token_used
