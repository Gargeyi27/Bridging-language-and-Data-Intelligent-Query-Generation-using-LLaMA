# Bridging-language-and-Data-Intelligent-Query-Generation-using-LLaMA
This project implements an AI-powered SQL agent that allows users to query databases using plain English. The system leverages TinyLLaMA (or DialoGPT as a fallback) with LangChain to generate SQL queries from natural language inputs and visualizes the results, making data more accessible for non-technical users.
# Install compatible packages
!pip install -q langchain langchain-experimental transformers torch sqlalchemy matplotlib pandas

# Import necessary libraries
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import re
import time
import os

# Remove existing database file if it exists
if os.path.exists('sample.db'):
    os.remove('sample.db')

# Create a sample database with some data
def create_sample_database():
    # Create SQLite database
    conn = sqlite3.connect('sample.db')
    cursor = conn.cursor()
    
    # Create employees table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            salary INTEGER NOT NULL,
            hire_date TEXT NOT NULL
        )
    ''')
    
    # Insert sample data
    employees_data = [
        (1, 'John Smith', 'Sales', 50000, '2020-01-15'),
        (2, 'Jane Doe', 'Marketing', 60000, '2019-03-23'),
        (3, 'Mike Johnson', 'Sales', 55000, '2021-06-10'),
        (4, 'Sarah Williams', 'HR', 65000, '2018-11-05'),
        (5, 'David Brown', 'Marketing', 70000, '2020-09-12'),
        (6, 'Emily Davis', 'Engineering', 85000, '2019-07-30'),
        (7, 'Robert Wilson', 'Engineering', 90000, '2020-03-18'),
        (8, 'Lisa Miller', 'HR', 60000, '2021-01-22')
    ]
    
    cursor.executemany('INSERT OR IGNORE INTO employees VALUES (?, ?, ?, ?, ?)', employees_data)
    
    # Create sales table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sales (
            id INTEGER PRIMARY KEY,
            employee_id INTEGER,
            sale_amount INTEGER NOT NULL,
            sale_date TEXT NOT NULL,
            FOREIGN KEY (employee_id) REFERENCES employees (id)
        )
    ''')
    
    # Insert sample sales data
    sales_data = [
        (1, 1, 10000, '2023-01-05'),
        (2, 1, 15000, '2023-01-12'),
        (3, 2, 8000, '2023-01-08'),
        (4, 3, 12000, '2023-01-10'),
        (5, 3, 9000, '2023-01-15'),
        (6, 5, 20000, '2023-01-20'),
        (7, 5, 18000, '2023-01-25')
    ]
    
    cursor.executemany('INSERT OR IGNORE INTO sales VALUES (?, ?, ?, ?)', sales_data)
    conn.commit()
    conn.close()
    
    print("Sample database created successfully!")

# Create the sample database
create_sample_database()

# Import LangChain components after database creation
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.chains import create_sql_query_chain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Initialize the database connection
db = SQLDatabase.from_uri("sqlite:///sample.db")

# Use a smaller, faster model that works well in Colab
model_name = "microsoft/DialoGPT-small"
print("Loading model...")

try:
    # Load model with optimizations
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True
    )
    
    # Create a text generation pipeline with optimizations
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=64,
        temperature=0.1,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    # Create the LLM instance
    llm = HuggingFacePipeline(pipeline=pipe)
    
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Falling back to rule-based SQL generation")
    llm = None

# Create a custom prompt for SQL generation
prompt_template = """
You are a SQL expert. Given an input question, create a syntactically correct SQL query to run.
Only use the following tables:
{table_info}

Question: {input}
SQLQuery:
"""

prompt = PromptTemplate(
    input_variables=["input", "table_info"],
    template=prompt_template
)

# Create the SQL query chain if model is available
if llm:
    try:
        sql_chain = create_sql_query_chain(llm, db, prompt=prompt)
        print("SQL chain created successfully!")
    except Exception as e:
        print(f"Error creating SQL chain: {e}")
        llm = None

# Cache for storing previous queries to avoid reprocessing
query_cache = {}

# Predefined queries for common questions to speed up response time
predefined_queries = {
    "show me all employees in the sales department": "SELECT name, department FROM employees WHERE department = 'Sales'",
    "what is the average salary by department": "SELECT department, AVG(salary) as avg_salary FROM employees GROUP BY department",
    "who has the highest salary": "SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 1",
    "show me total sales by employee": "SELECT e.name, SUM(s.sale_amount) as total_sales FROM employees e JOIN sales s ON e.id = s.employee_id GROUP BY e.name",
    "which department has the most employees": "SELECT department, COUNT(*) as employee_count FROM employees GROUP BY department ORDER BY employee_count DESC LIMIT 1",
    "show all employees": "SELECT name, department, salary FROM employees LIMIT 10",
    "show sales data": "SELECT e.name, s.sale_amount, s.sale_date FROM sales s JOIN employees e ON s.employee_id = e.id LIMIT 10"
}

# Rule-based SQL generator for fallback
def rule_based_sql_generator(question):
    question_lower = question.lower()
    
    if "sales department" in question_lower or "sales employees" in question_lower:
        return "SELECT name, department FROM employees WHERE department = 'Sales'"
    elif "marketing department" in question_lower or "marketing employees" in question_lower:
        return "SELECT name, department FROM employees WHERE department = 'Marketing'"
    elif "hr department" in question_lower or "hr employees" in question_lower:
        return "SELECT name, department FROM employees WHERE department = 'HR'"
    elif "engineering department" in question_lower or "engineering employees" in question_lower:
        return "SELECT name, department FROM employees WHERE department = 'Engineering'"
    elif "average salary" in question_lower:
        return "SELECT department, AVG(salary) as avg_salary FROM employees GROUP BY department"
    elif "highest salary" in question_lower:
        return "SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 1"
    elif "total sales" in question_lower:
        return "SELECT e.name, SUM(s.sale_amount) as total_sales FROM employees e JOIN sales s ON e.id = s.employee_id GROUP BY e.name"
    elif "most employees" in question_lower:
        return "SELECT department, COUNT(*) as employee_count FROM employees GROUP BY department ORDER BY employee_count DESC LIMIT 1"
    elif "all employees" in question_lower:
        return "SELECT name, department, salary FROM employees LIMIT 10"
    elif "sales data" in question_lower:
        return "SELECT e.name, s.sale_amount, s.sale_date FROM sales s JOIN employees e ON s.employee_id = e.id LIMIT 10"
    else:
        return None

# Function to execute the SQL query and return results
def run_sql_query(sql_query):
    try:
        # Clean up the SQL query
        sql_query = re.sub(r'```sql|```', '', sql_query).strip()
        
        # Execute the query
        conn = sqlite3.connect('sample.db')
        df = pd.read_sql_query(sql_query, conn)
        conn.close()
        
        return df
    except Exception as e:
        return f"Error executing query: {str(e)}"

# Function to process natural language query
def process_query(question):
    start_time = time.time()
    print(f"Question: {question}")
    
    # Check cache first
    if question.lower() in query_cache:
        print("Using cached result...")
        sql_query, result = query_cache[question.lower()]
        print(f"Generated SQL: {sql_query}")
        end_time = time.time()
        print(f"Query processed in {end_time - start_time:.2f} seconds")
        return sql_query, result
    
    # Check predefined queries
    if question.lower() in predefined_queries:
        print("Using predefined query...")
        sql_query = predefined_queries[question.lower()]
        result = run_sql_query(sql_query)
        print(f"SQL: {sql_query}")
        
        # Cache the result
        query_cache[question.lower()] = (sql_query, result)
        
        end_time = time.time()
        print(f"Query processed in {end_time - start_time:.2f} seconds")
        return sql_query, result
    
    # Try rule-based approach
    rule_based_sql = rule_based_sql_generator(question)
    if rule_based_sql:
        print("Using rule-based query...")
        result = run_sql_query(rule_based_sql)
        print(f"SQL: {rule_based_sql}")
        
        # Cache the result
        query_cache[question.lower()] = (rule_based_sql, result)
        
        end_time = time.time()
        print(f"Query processed in {end_time - start_time:.2f} seconds")
        return rule_based_sql, result
    
    # Fall back to LLM if available
    if llm:
        try:
            print("Using LLM to generate query...")
            sql_query = sql_chain.invoke({"question": question})
            result = run_sql_query(sql_query)
            print(f"Generated SQL: {sql_query}")
            
            # Cache the result
            query_cache[question.lower()] = (sql_query, result)
            
            end_time = time.time()
            print(f"Query processed in {end_time - start_time:.2f} seconds")
            return sql_query, result
        except Exception as e:
            print(f"Error with LLM: {e}")
    
    # Final fallback
    print("Could not generate SQL for this question. Try a different question.")
    return None, None

# Function to visualize results
def visualize_results(result, question):
    try:
        if result is None or isinstance(result, str) and "Error" in result:
            print("Cannot visualize due to error in query execution")
            return
        
        if isinstance(result, pd.DataFrame):
            df = result
        else:
            # Try to convert to DataFrame
            df = pd.DataFrame(result)
        
        if df.empty:
            print("No data to visualize")
            return
        
        # Simple visualization based on the data shape and question content
        if len(df.columns) >= 2:
            # Bar chart for two-column data
            plt.figure(figsize=(10, 6))
            plt.bar(df.iloc[:, 0].astype(str), df.iloc[:, 1].astype(float))
            plt.title(question)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        elif len(df.columns) == 1:
            # Pie chart for single column categorical data
            value_counts = df.iloc[:, 0].value_counts()
            plt.figure(figsize=(8, 8))
            plt.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%')
            plt.title(question)
            plt.show()
        else:
            # Default to table display
            print("Tabular results:")
            print(df.to_string(index=False))
            
    except Exception as e:
        print(f"Visualization error: {str(e)}")
        print("Raw result:")
        print(result)

# Main function to run the SQL agent
def sql_agent():
    print("Welcome to the Natural Language SQL Query Agent!")
    print("You can ask questions about the employee and sales data in plain English.")
    print("Example questions:")
    print("- Show me all employees in the Sales department")
    print("- What is the average salary by department?")
    print("- Who has the highest salary?")
    print("- Show me total sales by employee")
    print("- Type 'exit' to quit\n")
    
    # Pre-warm with a simple query
    print("Initializing system...")
    try:
        warmup_query = "SELECT COUNT(*) FROM employees"
        run_sql_query(warmup_query)
        print("System initialized successfully!\n")
    except Exception as e:
        print(f"Initialization completed with minor issues: {e}\n")
    
    # Now let the user ask questions
    while True:
        question = input("\nEnter your question (or 'exit' to quit): ")
        
        if question.lower() == 'exit':
            print("Goodbye!")
            break
            
        try:
            sql_query, result = process_query(question)
            if result is not None:
                visualize_results(result, question)
        except Exception as e:
            print(f"An error occurred: {str(e)}")

# Run the agent
if __name__ == "__main__":
    sql_agent()
