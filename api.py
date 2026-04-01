import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config


class NebulaGraphQA:
    def __init__(self):
        # 🔹 Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        # 🔹 Initialize DB
        self.connection_pool = self._init_connection()

    def _init_connection(self):
        host = os.getenv("NEBULA_HOST", "10.2.2.22")
        port = int(os.getenv("NEBULA_PORT", "9669"))

        config = Config()
        config.max_connection_pool_size = 10

        pool = ConnectionPool()
        ok = pool.init([(host, port)], config)

        if not ok:
            raise Exception("Failed to connect to Nebula Graph")

        return pool

    # 🔹 Execute NGQL
    def run_query(self, query: str):
        user = os.getenv("NEBULA_USER", "root")
        password = os.getenv("NEBULA_PASSWORD", "nebula")
        space = os.getenv("NEBULA_SPACE", "financial")
        session = self.connection_pool.get_session(user, password)

        try:
            session.execute(f"USE {space};")
            result = session.execute(query)
            return result
        finally:
            session.release()

    # 🔹 Convert NL → NGQL
    def generate_ngql(self, question: str):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in Nebula Graph Query Language (NGQL).

Convert the user question into a valid NGQL query.

Schema:
- TAG: Person (no properties; vertex ID is the person's name)
- EDGE: Transaction(amount int) — directed from borrower to lender

Rules:
- Use the vertex ID (id(v)) to match person names, not a property
- Only valid NGQL syntax
- Return ONLY the query, no explanation

Example:
MATCH (v:Person)-[e:Transaction]->(u:Person)
WHERE id(v) == "Arjun Sharma"
RETURN id(u) AS lender, e.Transaction.amount AS amount;
"""),
            ("human", f"Question: {question}")
        ])

        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({})

    # 🔹 Convert result → final answer
    def generate_answer(self, question, query, db_result):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant.

Given:
- user question
- graph query
- database result

Generate a clear final answer.
"""),
            ("human", f"""
Question: {question}

Query:
{query}

Result:
{db_result}

Answer:
""")
        ])

        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({})

    # 🔹 Main pipeline
    def ask(self, question: str):
        # Step 1: NL → NGQL
        ngql = self.generate_ngql(question)

        # Step 2: Execute query
        result = self.run_query(ngql)

        try:
            db_data = result.as_primitive()
        except:
            db_data = str(result)

        # Step 3: Final answer
        answer = self.generate_answer(
            question,
            ngql,
            json.dumps(db_data, indent=2)
        )

        return {
            "query": ngql,
            "raw_result": db_data,
            "answer": answer
        }