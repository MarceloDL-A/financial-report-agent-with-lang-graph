import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import OpenAI
import json
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
import pandas as pd
from io import StringIO

memory = SqliteSaver.from_conn_string(":memory:")

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
tavily = os.getenv("TAVILY_API_KEY")

# llm_name = "gpt-3.5-turbo"
llm_name = "gpt-4o"
model = ChatOpenAI(api_key=openai_key, model=llm_name)

from tavily import TavilyClient
tavily = TavilyClient(api_key=tavily)

from typing import TypedDict, List
from langchain_core.pydantic_v1 import BaseModel

class AgentState(TypedDict):
    task: str
    competitors: List[str]
    csv_file: str
    financial_data: str
    analysis: str
    competitor_data: str
    comparison: str
    feedback: str
    report: str
    content: List[str]
    revision_number: int
    max_revisions: int

class Queries(BaseModel):
    queries: List[str]

# Definir os prompts para cada nó - MELHORE SE NECESSÁRIO
GATHER_FINANCIALS_PROMPT = """Você é um analista financeiro especializado. Coleta os dados financeiros para a empresa dada. Forneça dados financeiros detalhados."""
ANALYZE_DATA_PROMPT = """Você é um analista financeiro especializado. Analise os dados financeiros fornecidos e forneça insights e análises detalhadas."""
RESEARCH_COMPETITORS_PROMPT = """Você é um pesquisador encarregado de fornecer informações sobre empresas similares para comparação de desempenho. Gere uma lista de consultas de pesquisa para coletar informações relevantes. Gere no máximo 3 consultas."""
COMPETE_PERFORMANCE_PROMPT = """Você é um analista financeiro especializado. Compare o desempenho financeiro da empresa dada com seus concorrentes com base nos dados fornecidos. **CERTIFIQUE-SE DE INCLUIR OS NOMES DOS CONCORRENTES NA COMPARAÇÃO.**"""
FEEDBACK_PROMPT = """Você é um revisor. Forneça feedback detalhado e críticas para o relatório de comparação financeira fornecido. Inclua quaisquer informações adicionais ou revisões necessárias."""
WRITE_REPORT_PROMPT = """Você é um redator de relatórios financeiros. Escreva um relatório financeiro abrangente com base na análise, pesquisa de concorrentes, comparação e feedback fornecidos."""
RESEARCH_CRITIQUE_PROMPT = """Você é um pesquisador encarregado de fornecer informações para responder à crítica fornecida. Gere uma lista de consultas de pesquisa para coletar informações relevantes. Gere no máximo 3 consultas."""

def gather_financials_node(state: AgentState):
    # Ler o arquivo CSV em um DataFrame do pandas
    csv_file = state["csv_file"]
    df = pd.read_csv(StringIO(csv_file))

    # Converter o DataFrame para uma string
    financial_data_str = df.to_string(index=False)

    # Combinar a string de dados financeiros com a tarefa
    combined_content = (
        f"{state['task']}\n\nAqui estão os dados financeiros:\n\n{financial_data_str}"
    )

    messages = [
        SystemMessage(content=GATHER_FINANCIALS_PROMPT),
        HumanMessage(content=combined_content),
    ]

    response = model.invoke(messages)
    return {"financial_data": response.content}

def analyze_data_node(state: AgentState):
    messages = [
        SystemMessage(content=ANALYZE_DATA_PROMPT),
        HumanMessage(content=state["financial_data"]),
    ]
    response = model.invoke(messages)
    return {"analysis": response.content}

def research_competitors_node(state: AgentState):
    content = state["content"] or []
    for competitor in state["competitors"]:
        queries = model.with_structured_output(Queries).invoke(
            [
                SystemMessage(content=RESEARCH_COMPETITORS_PROMPT),
                HumanMessage(content=competitor),
            ]
        )
        for q in queries.queries:
            response = tavily.search(query=q, max_results=2)
            for r in response["results"]:
                content.append(r["content"])
    return {"content": content}

def compare_performance_node(state: AgentState):
    content = "\n\n".join(state["content"] or [])
    user_message = HumanMessage(
        content=f"{state['task']}\n\nAqui está a análise financeira:\n\n{state['analysis']}"
    )
    messages = [
        SystemMessage(content=COMPETE_PERFORMANCE_PROMPT.format(content=content)),
        user_message,
    ]
    response = model.invoke(messages)
    return {
        "comparison": response.content,
        "revision_number": state.get("revision_number", 1) + 1,
    }

def research_critique_node(state: AgentState):
    queries = model.with_structured_output(Queries).invoke(
        [
            SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
            HumanMessage(content=state["feedback"]),
        ]
    )
    content = state["content"] or []
    for q in queries.queries:
        response = tavily.search(query=q, max_results=2)
        for r in response["results"]:
            content.append(r["content"])
    return {"content": content}

def collect_feedback_node(state: AgentState):
    messages = [
        SystemMessage(content=FEEDBACK_PROMPT),
        HumanMessage(content=state["comparison"]),
    ]
    response = model.invoke(messages)
    return {"feedback": response.content}

def write_report_node(state: AgentState):
    messages = [
        SystemMessage(content=WRITE_REPORT_PROMPT),
        HumanMessage(content=state["comparison"]),
    ]
    response = model.invoke(messages)
    return {"report": response.content}

def should_continue(state):
    if state["revision_number"] > state["max_revisions"]:
        return END
    return "collect_feedback"

builder = StateGraph(AgentState)

builder.add_node("gather_financials", gather_financials_node)
builder.add_node("analyze_data", analyze_data_node)
builder.add_node("research_competitors", research_competitors_node)
builder.add_node("compare_performance", compare_performance_node)
builder.add_node("collect_feedback", collect_feedback_node)
builder.add_node("research_critique", research_critique_node)

builder.add_node("write_report", write_report_node)

builder.set_entry_point("gather_financials")

builder.add_conditional_edges(
    "compare_performance",
    should_continue,
    {END: END, "collect_feedback": "collect_feedback"},
)

builder.add_edge("gather_financials", "analyze_data")
builder.add_edge("analyze_data", "research_competitors")
builder.add_edge("research_competitors", "compare_performance")
builder.add_edge("collect_feedback", "research_critique")
builder.add_edge("research_critique", "compare_performance")
builder.add_edge("compare_performance", "write_report")

graph = builder.compile(checkpointer=memory)

# ==== Streamlit UI ====
import streamlit as st

def main():
    st.title("Agente de Relatórios de Desempenho Financeiro")

    st.markdown("""
    ## Instruções:
    1. **Digite a tarefa específica** que deseja analisar.
    2. **Insira os nomes dos concorrentes**, um por linha.
    3. **Faça o upload do arquivo CSV** contendo os dados financeiros da empresa.
    4. Clique em **"Iniciar Análise"** para gerar o relatório.
    """)

    task = st.text_input(
        "Digite a tarefa:",
        "Analise o desempenho financeiro de nossa empresa (MyAICo) em comparação com os concorrentes",
    )
    competitors = st.text_area("Digite os nomes dos concorrentes (um por linha):").split("\n")
    max_revisions = st.number_input("Máximo de Revisões", min_value=1, value=2)
    uploaded_file = st.file_uploader(
        "Envie um arquivo CSV com os dados financeiros da empresa", type=["csv"]
    )

    if st.button("Iniciar Análise") and uploaded_file is not None:
        # Ler o arquivo CSV enviado
        csv_data = uploaded_file.getvalue().decode("utf-8")

        initial_state = {
            "task": task,
            "competitors": [comp.strip() for comp in competitors if comp.strip()],
            "csv_file": csv_data,
            "max_revisions": max_revisions,
            "revision_number": 1,
        }
        thread = {"configurable": {"thread_id": "1"}}

        final_state = None
        for s in graph.stream(initial_state, thread):
            st.markdown("### Resultado da Análise")
            if "gather_financials" in s:
                financial_data = s["gather_financials"]["financial_data"]
                st.markdown("#### Dados Financeiros Coletados")
                st.markdown(financial_data)

            if "analyze_data" in s:
                analysis = s["analyze_data"]["analysis"]
                st.markdown("#### Análise dos Dados Financeiros")
                st.markdown(analysis)

            if "research_competitors" in s:
                competitors_data = s["research_competitors"]["content"]
                st.markdown("#### Pesquisa sobre Concorrentes")
                for i, content in enumerate(competitors_data):
                    st.markdown(f"**Concorrente {i + 1}:** {content}")

            if "compare_performance" in s:
                comparison = s["compare_performance"]["comparison"]
                st.markdown("#### Comparação de Desempenho")
                st.markdown(comparison)

            if "collect_feedback" in s:
                feedback = s["collect_feedback"]["feedback"]
                st.markdown("#### Feedback sobre a Comparação")
                st.markdown(feedback)

            if "write_report" in s:
                report = s["write_report"]["report"]
                st.markdown("## Relatório Final")
                st.markdown(report)
            
            final_state = s

if __name__ == "__main__":
    main()
# ==== Fim do Streamlit UI ====

