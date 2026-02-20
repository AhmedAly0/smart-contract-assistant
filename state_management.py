"""
Structured extraction via LLM (RExtract).
"""

import re

from langchain.schema.runnable.passthrough import RunnableAssign
from langchain.output_parsers import PydanticOutputParser


def RExtract(pydantic_class, llm, prompt):
    """Create a Runnable that extracts structured data from text using an LLM.

    1. Generate format_instructions from the Pydantic model
    2. Inject them into the prompt
    3. Send to LLM
    4. Parse the output back into a Pydantic instance
    """
    parser = PydanticOutputParser(pydantic_object=pydantic_class)

    instruct_merge = RunnableAssign({
        "format_instructions": lambda x: parser.get_format_instructions()
    })

    def preparse(string):
        if isinstance(string, str):
            string = re.sub(r"```json\s*", "", string)
            string = re.sub(r"```\s*", "", string)
            match = re.search(r"\{.*\}", string, re.DOTALL)
            if match:
                string = match.group()
        return string

    return instruct_merge | prompt | llm | preparse | parser
