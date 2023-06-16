import openai
import re
from approaches.approach import Approach
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
from langchain.llms.openai import AzureOpenAI
from langchain.prompts import PromptTemplate, BasePromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.agents import Tool, AgentExecutor
from langchain.agents.react.base import ReActDocstoreAgent
from langchainadapters import HtmlCallbackHandler
from text import nonewlines
from typing import List

class ReadDecomposeAsk(Approach):
    def __init__(self, search_client: SearchClient, openai_deployment: str, sourcepage_field: str, content_field: str):
        self.search_client = search_client
        self.openai_deployment = openai_deployment
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field

    def search(self, q: str, overrides: dict) -> str:
        use_semantic_captions = True if overrides.get("semantic_captions") else False
        top = overrides.get("top") or 3
        exclude_category = overrides.get("exclude_category") or None
        filter = "category ne '{}'".format(exclude_category.replace("'", "''")) if exclude_category else None

        if overrides.get("semantic_ranker"):
            r = self.search_client.search(q,
                                          filter=filter,
                                          query_type=QueryType.SEMANTIC, 
                                          query_language="en-us", 
                                          query_speller="lexicon", 
                                          semantic_configuration_name="default", 
                                          top = top,
                                          query_caption="extractive|highlight-false" if use_semantic_captions else None)
        else:
            r = self.search_client.search(q, filter=filter, top=top)
        if use_semantic_captions:
            self.results = [doc[self.sourcepage_field] + ":" + nonewlines(" . ".join([c.text for c in doc['@search.captions'] ])) for doc in r]
        else:
            self.results = [doc[self.sourcepage_field] + ":" + nonewlines(doc[self.content_field][:500]) for doc in r]
        return "\n".join(self.results)

    def lookup(self, q: str) -> str:
        r = self.search_client.search(q,
                                      top = 1,
                                      include_total_count=True,
                                      query_type=QueryType.SEMANTIC, 
                                      query_language="en-us", 
                                      query_speller="lexicon", 
                                      semantic_configuration_name="default",
                                      query_answer="extractive|count-1",
                                      query_caption="extractive|highlight-false")
        
        answers = r.get_answers()
        if answers and len(answers) > 0:
            return answers[0].text
        if r.get_count() > 0:
            return "\n".join(d['content'] for d in r)
        return None        

    def run(self, q: str, overrides: dict) -> any:
        # Not great to keep this as instance state, won't work with interleaving (e.g. if using async), but keeps the example simple
        self.results = None

        # Use to capture thought process during iterations
        cb_handler = HtmlCallbackHandler()
        cb_manager = CallbackManager(handlers=[cb_handler])

        llm = AzureOpenAI(deployment_name=self.openai_deployment, temperature=overrides.get("temperature") or 0.3, openai_api_key=openai.api_key)
        tools = [
            Tool(name="Search", func=lambda q: self.search(q, overrides), description="useful for when you need to ask with search", callbacks=cb_manager),
            Tool(name="Lookup", func=self.lookup, description="useful for when you need to ask with lookup", callbacks=cb_manager)
        ]

        # Like results above, not great to keep this as a global, will interfere with interleaving
        global prompt
        prompt_prefix = overrides.get("prompt_template")
        prompt = PromptTemplate.from_examples(
            EXAMPLES, SUFFIX, ["input", "agent_scratchpad"], prompt_prefix + "\n\n" + PREFIX if prompt_prefix else PREFIX)

        agent = ReAct.from_llm_and_tools(llm, tools)
        chain = AgentExecutor.from_agent_and_tools(agent, tools, verbose=True, callback_manager=cb_manager)
        result = chain.run(q)

        # Replace substrings of the form <file.ext> with [file.ext] so that the frontend can render them as links, match them with a regex to avoid 
        # generalizing too much and disrupt HTML snippets if present
        result = re.sub(r"<([a-zA-Z0-9_ \-\.]+)>", r"[\1]", result)

        return {"data_points": self.results or [], "answer": result, "thoughts": cb_handler.get_and_reset_log()}
    
class ReAct(ReActDocstoreAgent):
    @classmethod
    def create_prompt(cls, tools: List[Tool]) -> BasePromptTemplate:
        return prompt
    
# Modified version of langchain's ReAct prompt that includes instructions and examples for how to cite information sources
EXAMPLES = [
    """Question: Which type of data should be sent from video cameras in a native binary format?
Select only one answer.
A) structured
B) semi-structured
C) unstructured
Answer: Video data should be sent from video cameras in a native binary format. As mentioned in source 1, binary data is used to store images, video, audio, and application-specific documents. Video data is a type of unstructured data that is typically stored in its native binary format.
Citations:
1. \[Explore file storage - Training _ Microsoft Learn-2.pdf\]""",
]
SUFFIX = """\nQuestion: {input}
{agent_scratchpad}"""
PREFIX = "You are my test proctor. Test me on Azure Data Fundamentals also known as DP-900. Ask one multiple choice question and allow me to answer before asking the next question. If the answer is wrong, provide the correct answer and ask the next question. Answer ONLY with the facts listed in the list of sources below. If there isn't enough information below, say you don't know. Do not generate answers that don't use the sources below. If asking a clarifying question to the user would help, ask the question. " \
"Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response. Use square brakets to reference the source, e.g. [info1.txt]. Don't combine sources, list each source separately, e.g. [info1.txt][info2.pdf]. "
