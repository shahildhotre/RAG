import os
from typing import List, Dict
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from neo4j import GraphDatabase
import ast
import re
from tenacity import retry, stop_after_attempt, wait_exponential

class CodebaseRAGBot:
    def __init__(self, codebase_path: str):
        load_dotenv()
        
        # Get HuggingFace API token
        self.hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
        if not self.hf_token:
            raise ValueError("HuggingFace API token is required")
        
        # Neo4j connection
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        self.graph_db = GraphDatabase.driver(
            self.neo4j_uri, 
            auth=(self.neo4j_user, self.neo4j_password)
        )

        # Initialize components
        self.documents = self.load_codebase(codebase_path)
        self.vector_store = self.setup_vector_store()
        self.conversation_chain = self.setup_conversation_chain()
        
        # Build code graph
        self.build_code_graph(codebase_path)
    
    def build_code_graph(self, codebase_path: str):
        """Build a graph representation of the codebase in Neo4j"""
        with self.graph_db.session() as session:
            # Clear existing graph
            session.run("MATCH (n) DETACH DELETE n")
            
            # Walk through the codebase
            for root, _, files in os.walk(codebase_path):
                for file in files:
                    if file.endswith('.py'):  # Start with Python files
                        file_path = os.path.join(root, file)
                        self._process_python_file(session, file_path)

    def _process_python_file(self, session, file_path: str):
        """Process a Python file and create nodes and relationships in Neo4j"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
            # Create file node
            file_name = os.path.basename(file_path)
            session.run("""
                CREATE (f:File {name: $name, path: $path})
            """, name=file_name, path=file_path)
            
            # Parse Python code
            tree = ast.parse(content)
            
            # Process classes and functions
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Create class node
                    session.run("""
                        MATCH (f:File {path: $file_path})
                        CREATE (c:Class {name: $name, docstring: $docstring})
                        CREATE (f)-[:CONTAINS]->(c)
                    """, file_path=file_path, 
                        name=node.name,
                        docstring=ast.get_docstring(node) or '')
                    
                elif isinstance(node, ast.FunctionDef):
                    # Create function node
                    session.run("""
                        MATCH (f:File {path: $file_path})
                        CREATE (func:Function {
                            name: $name,
                            docstring: $docstring,
                            args: $args
                        })
                        CREATE (f)-[:CONTAINS]->(func)
                    """, file_path=file_path,
                        name=node.name,
                        docstring=ast.get_docstring(node) or '',
                        args=', '.join([arg.arg for arg in node.args.args]))
                    
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

    def query_graph(self, query: str) -> str:
        """Query the Neo4j graph for code relationships"""
        with self.graph_db.session() as session:
            if "class" in query.lower():
                result = session.run("""
                    MATCH (c:Class)
                    RETURN c.name, c.docstring
                """)
                classes = [f"Class: {record['c.name']}\nDocstring: {record['c.docstring']}"
                          for record in result]
                return "\n\n".join(classes)
            
            elif "function" in query.lower():
                result = session.run("""
                    MATCH (f:Function)
                    RETURN f.name, f.docstring, f.args
                """)
                functions = [f"Function: {record['f.name']}\nArgs: {record['f.args']}\nDocstring: {record['f.docstring']}"
                           for record in result]
                return "\n\n".join(functions)
            
            return None

    def load_codebase(self, codebase_path: str) -> List:
        """Load and preprocess code files"""
        code_extensions = [
            ".py", ".js", ".java", ".cpp", ".h", ".cs", ".rb", 
            ".php", ".go", ".rs", ".ts", ".html", ".css"
        ]
        
        # Create a loader that filters for code files
        loader = DirectoryLoader(
            codebase_path,
            glob="**/*.*",
            loader_cls=TextLoader,
            loader_kwargs={'autodetect_encoding': True},
            show_progress=True,
            use_multithreading=True,
        )
        documents = loader.load()
        
        # Use a code-aware text splitter
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=2000,  # Increased chunk size for better context
            chunk_overlap=200,
            length_function=len,
        )
        
        # Remove duplicate content before splitting
        unique_content = set()
        filtered_docs = []
        for doc in documents:
            if doc.page_content not in unique_content:
                unique_content.add(doc.page_content)
                filtered_docs.append(doc)
        
        return splitter.split_documents(filtered_docs)
    
    def setup_vector_store(self):
        """Setup vector store with HuggingFace embeddings"""
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        return Chroma.from_documents(
            documents=self.documents,
            embedding=embeddings,
            persist_directory="./code_knowledge_base"
        )
    
    def setup_conversation_chain(self):
        """Setup the conversation chain with HuggingFace model"""
        # Option 1: Use a smaller, faster model
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-base",  # Smaller model
            model_kwargs={"temperature": 0.5, "max_length": 512},  # Reduced max_length
            huggingfacehub_api_token=self.hf_token
        )
        
        # Option 2: Add retry logic
        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
        def get_llm_response(*args, **kwargs):
            return llm(*args, **kwargs)
        
        # Create a more specific prompt template for code analysis
        CUSTOM_PROMPT = """You are an expert code analyzer. When analyzing code:
        
        - For "Major components" questions:
          - List the main classes, modules, and key functionalities
          - Explain the purpose of each component
          - Ignore duplicate code
        
        - For function-related questions:
          - List all unique functions
          - Show their arguments and return types
          - Provide a brief description of each function
        
        Current context: {context}
        Chat history: {chat_history}
        Question: {question}
        
        Please analyze the code and provide a clear, structured response:"""
        
        prompt = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template=CUSTOM_PROMPT
        )
        
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True
        )
        
        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 4}
            ),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": prompt}
        )
    
    def get_graph_data(self) -> Dict:
        """Return graph data in a format suitable for visualization"""
        with self.graph_db.session() as session:
            # Get nodes
            nodes_result = session.run("""
                MATCH (n)
                RETURN DISTINCT
                    id(n) as id,
                    labels(n) as labels,
                    properties(n) as properties
            """)
            nodes = [
                {
                    "id": record["id"],
                    "label": record["labels"][0],  # Using first label
                    "properties": record["properties"]
                }
                for record in nodes_result
            ]

            # Get relationships
            edges_result = session.run("""
                MATCH (source)-[r]->(target)
                RETURN DISTINCT
                    id(source) as source_id,
                    id(target) as target_id,
                    type(r) as relationship_type
            """)
            edges = [
                {
                    "source": record["source_id"],
                    "target": record["target_id"],
                    "label": record["relationship_type"]
                }
                for record in edges_result
            ]

            return {
                "nodes": nodes,
                "edges": edges
            }

    def chat(self, query: str) -> Dict:
        """Process a query and return the response with graph data"""
        try:
            # First try to get information from the graph
            graph_response = self.query_graph(query)
            
            try:
                if graph_response:
                    # Combine graph information with LLM response
                    enhanced_query = f"{query}\nAdditional context from code structure:\n{graph_response}"
                    response = self.conversation_chain.invoke({"question": enhanced_query})
                else:
                    # Fall back to regular RAG response
                    response = self.conversation_chain.invoke({"question": query})
            except Exception as llm_error:
                # Fallback response if LLM fails
                return {
                    "answer": f"Graph Analysis: {graph_response}\n\nNote: LLM response unavailable due to service issues.",
                    "graph_data": self.get_graph_data()
                }
                
            # Include graph data in response
            return {
                "answer": response["answer"],
                "graph_data": self.get_graph_data()
            }
        except Exception as e:
            return {
                "answer": f"Error processing query: {str(e)}",
                "graph_data": {}
            }
            response = self.conversation_chain.invoke({"question": query})
        
        # Include graph data in response
        return {
            "answer": response["answer"],
            "graph_data": self.get_graph_data()
        }

    def close(self):
        """Close the Neo4j connection"""
        self.graph_db.close()

def main():
    # Get the codebase path from user
    codebase_path = input("Enter the path to your codebase: ")
    
    if not os.path.exists(codebase_path):
        print(f"Error: Path '{codebase_path}' does not exist!")
        return
    
    print("\nInitializing CodebaseRAGBot... This may take a few minutes depending on the codebase size.")
    chatbot = CodebaseRAGBot(codebase_path)
    
    print("\nCodebaseRAGBot is ready! You can now ask questions about your code.")
    print("Example questions:")
    print("- What classes are defined in this codebase?")
    print("- Show me all functions and their arguments")
    print("- How are different components connected?")
    print("\nType 'quit' to exit.")
    
    try:
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() == 'quit':
                break
                
            try:
                response = chatbot.chat(user_input)
                print(f"\nBot: {response['answer']}")
            except Exception as e:
                print(f"\nError: {str(e)}")
    finally:
        chatbot.close()

if __name__ == "__main__":
    main() 