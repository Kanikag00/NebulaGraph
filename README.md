# Graph RAG System with Nebula Graph

This project implements a Retrieval-Augmented Generation (RAG) system on top of a graph database, enabling natural language queries to be translated into structured graph queries and returning meaningful insights.

---

## 🚀 Overview

The system allows users to query a directed graph database using natural language. It leverages an LLM to convert user queries into nGQL (Nebula Graph Query Language), executes them on the database, and reformats the results into human-readable responses.

---

## 🧠 Architecture

### Graph Database

- **Type:** Directed Graph  
- **Nodes:** ~1000 entities (represented by names)  
- **Edges:** Transactions between nodes  

Each edge represents a directed relationship from a source node to a target node.

---

## ⚙️ Workflow

The system follows a 4-step pipeline:

### 1. User Query
The user provides a query in natural language.  

**Example:**  
> "Show transactions from Alice to Bob"

---

### 2. Query Translation (LLM → nGQL)
The input query is converted into nGQL using an LLM (currently OpenAI-based).

---

### 3. Graph Database Execution
- The generated nGQL query is executed on the Nebula Graph database  
- Relevant nodes and edges are retrieved  

---

### 4. Response Reframing
- The raw database response is processed and reformatted  
- A clean, human-readable answer is returned to the user  

---

## 🛠️ Tech Stack

- **Graph Database:** Nebula Graph  
- **Query Language:** nGQL  
- **LLM Integration:** OpenAI API (for query translation)  
- **Backend Logic:** Python *(update if different)*  

---

## 📌 Features

- Natural language → graph query conversion  
- Directed graph transaction modeling  
- Modular RAG pipeline  
- Scalable query handling  

---

## 🔮 Upcoming Improvements

### 1. On-Prem Model Deployment
- Replace API-based LLM with locally hosted models  
- Use LoRA (Low-Rank Adaptation) for fine-tuning  
- **Goal:** Reduce latency and improve accessibility under hardware constraints  

---

### 2. Multi-hop Query Support
Enable complex reasoning across multiple nodes.  

**Example:**  
> "Find connections between Alice and Charlie through intermediaries"  

This will allow deeper graph traversal and richer insights.
