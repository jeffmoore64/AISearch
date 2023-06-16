import openai
from approaches.approach import Approach
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
from text import nonewlines

# Simple retrieve-then-read implementation, using the Cognitive Search and OpenAI APIs directly. It first retrieves
# top documents from search, then constructs a prompt with them, and then uses OpenAI to generate an completion 
# (answer) with that prompt.
class RetrieveThenReadApproach(Approach):

    template = \
"Test me by Azure Data Fundamentals. Ask me one multiple choice question and allow me to answer before asking the next question. " + \
"Answer ONLY with the facts listed in the list of sources below. If there isn't enough information below, say you don't know. " + \
"Do not generate answers that don't use the sources below. If asking a clarifying question to the user would help, ask the question. " + \
"Answer the question using only the data provided in the information sources below. " + \
"For tabular information return it as an html table. Do not return markdown format. " + \
"Each source has a name followed by colon and the actual data, quote the source name for each piece of data you use in the response. " + \
"For example, if the question is \"What color is the sky?\" and one of the information sources says \"info123: the sky is blue whenever it's not cloudy\", then answer with \"The sky is blue [info123]\" " \
"It's important to strictly follow the format where the name of the source is in square brackets at the end of the sentence, and only up to the prefix before the colon (\":\"). " + \
"If there are multiple sources, cite each one in their own square brackets. For example, use \"[info343][ref-76]\" and not \"[info343,ref-76]\". " + \
"Never quote tool names as sources." + \
"If you cannot answer using the sources below, say that you don't know. " + \
"\n\nYou can access to the following tools:" +\
"Use 'you' to refer to the individual asking the questions even if they ask with 'I'. " + \
"Answer the following question using only the data provided in the sources below. " + \
"For tabular information return it as an html table. Do not return markdown format. "  + \
"Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response. " + \
"If you cannot answer using the sources below, say you don't know. " + \
"""

###
Question: 'What is the deductible for the employee plan for a visit to Overlake in Bellevue?'

Sources:
info1.txt: deductibles depend on whether you are in-network or out-of-network. In-network deductibles are $500 for employee and $1000 for family. Out-of-network deductibles are $1000 for employee and $2000 for family.
info2.pdf: Overlake is in-network for the employee plan.
info3.pdf: Overlake is the name of the area that includes a park and ride near Bellevue.
info4.pdf: In-network institutions include Overlake, Swedish and others in the region

Answer:
In-network deductibles are $500 for employee and $1000 for family [info1.txt] and Overlake is in-network for the employee plan [info2.pdf][info4.pdf].

###
Question: '{q}'?

Sources:
{retrieved}

Answer:
"""

    def __init__(self, search_client: SearchClient, openai_deployment: str, sourcepage_field: str, content_field: str):
        self.search_client = search_client
        self.openai_deployment = openai_deployment
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field

    def run(self, q: str, overrides: dict) -> any:
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
                                          top=top, 
                                          query_caption="extractive|highlight-false" if use_semantic_captions else None)
        else:
            r = self.search_client.search(q, filter=filter, top=top)
        if use_semantic_captions:
            results = [doc[self.sourcepage_field] + ": " + nonewlines(" . ".join([c.text for c in doc['@search.captions']])) for doc in r]
        else:
            results = [doc[self.sourcepage_field] + ": " + nonewlines(doc[self.content_field]) for doc in r]
        content = "\n".join(results)

        prompt = (overrides.get("prompt_template") or self.template).format(q=q, retrieved=content)
        completion = openai.Completion.create(
            engine=self.openai_deployment, 
            prompt=prompt, 
            temperature=overrides.get("temperature") or 0.3, 
            max_tokens=1024, 
            n=1, 
            stop=["\n"])

        return {"data_points": results, "answer": completion.choices[0].text, "thoughts": f"Question:<br>{q}<br><br>Prompt:<br>" + prompt.replace('\n', '<br>')}
