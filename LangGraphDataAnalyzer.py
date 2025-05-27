import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from typing import Dict, List, Any, Tuple, Optional
import tempfile
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import traceback

# LangGraph imports
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.agents import Tool
from langchain_core.prompts import MessagesPlaceholder
import langchain.pydantic_v1

# Set environment variable for Groq API key
os.environ["GROQ_API_KEY"] = "Enter Your API Key"

# Set page config with error handling
try:
    st.set_page_config(
        page_title="Data Analysis AI Assistant",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
except Exception as e:
    st.error(f"Error setting page config: {str(e)}")

# Custom CSS
st.markdown("""
    <style>
        .main { padding: 2rem; }
        .sidebar .sidebar-content { padding: 1.5rem; }
        .stButton button {
            width: 100%;
            background-color: #4CAF50;
            color: white;
        }
        .stTextInput input { padding: 10px; }
        .stFileUploader { padding: 10px; }
        .stMarkdown {
            padding: 10px;
            border-radius: 5px;
            background-color: #f0f2f6;
        }
        .ai-response {
            padding: 15px;
            border-radius: 10px;
            background-color: #e6f7ff;
            margin-bottom: 15px;
        }
        .user-query {
            padding: 15px;
            border-radius: 10px;
            background-color: #f0f0f0;
            margin-bottom: 15px;
        }
        .stPlotlyChart {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
        }
        .tool-card {
            padding: 15px;
            border-radius: 8px;
            background-color: #f8f9fa;
            margin-bottom: 10px;
            border-left: 4px solid #4CAF50;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
    if "df" not in st.session_state:
        st.session_state.df = None
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None
    if "data_summary" not in st.session_state:
        st.session_state.data_summary = None
    if "selected_tools" not in st.session_state:
        st.session_state.selected_tools = []

# Sidebar
def render_sidebar():
    try:
        with st.sidebar:
            st.title("📊 Data Analysis Assistant")
            st.markdown("Upload your CSV file and describe your analysis goals.")
            
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            
            if uploaded_file is not None:
                st.session_state.uploaded_file = uploaded_file
                try:
                    if uploaded_file.name.endswith('.csv'):
                        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                        df = pd.read_csv(stringio)
                        st.session_state.df = df
                        st.success("File successfully loaded!")
                    else:
                        st.error("Please upload a CSV file.")
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
                    st.session_state.df = None
            
            st.markdown("---")
            st.markdown("### Analysis Settings")
            use_case = st.text_area("Describe your use case or analysis goals:", key="use_case_input")
            
            st.markdown("---")
            st.markdown("### About")
            st.markdown("""
            This AI assistant helps you analyze your data through:
            - Automated data profiling
            - Smart visualizations
            - Business insights generation
            - Conversational analysis
            """)
        return use_case
    except Exception as e:
        st.error(f"Sidebar rendering error: {str(e)}")
        return ""

# Initialize LLM
def initialize_llm():
    try:
        return ChatGroq(
            model="llama3-8b-8192",
            temperature=0.5,
            max_retries=3
        )
    except Exception as e:
        st.error(f"Failed to initialize LLM: {str(e)}")
        return None

# Enhanced data summary function with type conversion
def get_data_summary(df):
    try:
        # Attempt to convert columns to appropriate types
        for col in df.columns:
            # Try converting to numeric
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except:
                pass
            
            # Try converting to datetime
            try:
                df[col] = pd.to_datetime(df[col], errors='ignore')
            except:
                pass

        summary = f"""
        ### Dataset Overview
        - **Rows**: {len(df):,}
        - **Columns**: {len(df.columns)}
        - **Memory Usage**: {df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB
        
        ### Column Details
        """
        
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                summary += f"""
                **{col}** (Numeric):
                - Range: {df[col].min():.2f} to {df[col].max():.2f}
                - Mean: {df[col].mean():.2f} | Median: {df[col].median():.2f}
                - Std Dev: {df[col].std():.2f}
                - Missing: {df[col].isnull().sum()} ({df[col].isnull().mean() * 100:.1f}%)
                """
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                summary += f"""
                **{col}** (Datetime):
                - Range: {df[col].min()} to {df[col].max()}
                - Missing: {df[col].isnull().sum()} ({df[col].isnull().mean() * 100:.1f}%)
                """
            else:
                summary += f"""
                **{col}** (Categorical/Text):
                - Unique Values: {df[col].nunique()}
                - Top Value: "{df[col].mode()[0]}" ({(df[col] == df[col].mode()[0]).mean() * 100:.1f}%)
                - Missing: {df[col].isnull().sum()} ({df[col].isnull().mean() * 100:.1f}%)
                """
        
        # Correlation analysis
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 1:
            try:
                corr = df[numeric_cols].corr().abs()
                top_corr = corr.mask(np.triu(np.ones(corr.shape, dtype=bool)))
                top_corr_pairs = top_corr.stack().sort_values(ascending=False).head(3)
                
                summary += "\n### Top Correlations\n"
                for pair in top_corr_pairs.index:
                    summary += f"- {pair[0]} & {pair[1]}: {top_corr_pairs[pair]:.2f}\n"
            except Exception as e:
                summary += f"\n### Correlation Analysis\nError computing correlations: {str(e)}\n"
        
        return summary
    except Exception as e:
        return f"Error generating data summary: {str(e)}"

# Comprehensive Visualization Agent
def visualize_data(df, use_case=None):
    visualizations = []
    numeric_cols = df.select_dtypes(include=['number']).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    datetime_cols = df.select_dtypes(include=['datetime64']).columns

    try:
        # 1. Histograms for numeric columns
        for col in numeric_cols:
            try:
                fig = px.histogram(df, x=col, title=f'Distribution of {col}', nbins=50)
                visualizations.append(("histogram", col, fig))
            except Exception as e:
                st.warning(f"Could not create histogram for {col}: {str(e)}")

        # 2. Box plots for numeric columns
        for col in numeric_cols:
            try:
                fig = px.box(df, y=col, title=f'Box Plot of {col}')
                visualizations.append(("box", col, fig))
            except Exception as e:
                st.warning(f"Could not create box plot for {col}: {str(e)}")

        # 3. Correlation heatmap
        if len(numeric_cols) > 1:
            try:
                corr = df[numeric_cols].corr()
                fig = go.Figure(data=go.Heatmap(
                    z=corr.values,
                    x=corr.columns,
                    y=corr.index,
                    colorscale='RdBu',
                    zmin=-1,
                    zmax=1,
                    text=np.round(corr.values, 2),
                    texttemplate="%{text}"
                ))
                fig.update_layout(title='Correlation Heatmap')
                visualizations.append(("heatmap", "correlation", fig))
            except Exception as e:
                st.warning(f"Could not create correlation heatmap: {str(e)}")

        # 4. Scatter plots for numeric pairs
        if len(numeric_cols) >= 2:
            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    try:
                        fig = px.scatter(df, x=numeric_cols[i], y=numeric_cols[j], 
                                       title=f'{numeric_cols[i]} vs {numeric_cols[j]}')
                        visualizations.append(("scatter", f"{numeric_cols[i]}_vs_{numeric_cols[j]}", fig))
                    except Exception as e:
                        st.warning(f"Could not create scatter plot for {numeric_cols[i]} vs {numeric_cols[j]}: {str(e)}")

        # 5. Bar plots for categorical columns
        if len(cat_cols) > 0:
            for col in cat_cols:
                try:
                    value_counts = df[col].value_counts().head(10)
                    fig = px.bar(x=value_counts.index, y=value_counts.values,
                                title=f'Top 10 Categories in {col}',
                                labels={'x': col, 'y': 'Count'})
                    visualizations.append(("bar", col, fig))
                except Exception as e:
                    st.warning(f"Could not create bar plot for {col}: {str(e)}")

        # 6. Time series plots for datetime columns
        if len(datetime_cols) > 0 and len(numeric_cols) > 0:
            for dt_col in datetime_cols:
                for num_col in numeric_cols:
                    try:
                        fig = px.line(df.sort_values(dt_col), x=dt_col, y=num_col,
                                    title=f'{num_col} over Time')
                        visualizations.append(("line", f"{num_col}_over_time", fig))
                    except Exception as e:
                        st.warning(f"Could not create time series plot for {num_col}: {str(e)}")

        # 7. Categorical-numeric combinations
        if len(cat_cols) > 0 and len(numeric_cols) > 0:
            for cat_col in cat_cols[:3]:
                for num_col in numeric_cols[:3]:
                    try:
                        fig = px.box(df, x=cat_col, y=num_col,
                                   title=f'{num_col} by {cat_col}')
                        visualizations.append(("box", f"{num_col}_by_{cat_col}", fig))
                    except Exception as e:
                        st.warning(f"Could not create box plot for {num_col} by {cat_col}: {str(e)}")

        # 8. Pie chart for categorical columns
        if len(cat_cols) > 0:
            for col in cat_cols:
                try:
                    value_counts = df[col].value_counts().head(5)
                    fig = px.pie(values=value_counts.values, names=value_counts.index,
                                title=f'Top 5 Categories in {col}')
                    visualizations.append(("pie", col, fig))
                except Exception as e:
                    st.warning(f"Could not create pie chart for {col}: {str(e)}")

    except Exception as e:
        st.error(f"Error in visualization generation: {str(e)}")
        traceback.print_exc()

    return visualizations

# Use Case Reconstruction Agent
def reconstruct_use_case(df, analysis_results):
    try:
        llm = initialize_llm()
        if not llm:
            return "Failed to initialize LLM for use case reconstruction"

        prompt = """
        You are an expert data analyst tasked with reconstructing the user's use case based on:
        Data Summary: {data_summary}
        Analysis Results: {analysis_results}
        
        Generate a clear, concise use case description that explains:
        1. What type of data analysis the user likely wants
        2. The business context or problem they're trying to solve
        3. Key metrics or insights they're likely interested in
        4. Potential stakeholders
        
        Return the response in markdown format.
        """
        
        chain = PromptTemplate.from_template(prompt) | llm
        response = chain.invoke({
            "data_summary": get_data_summary(df),
            "analysis_results": analysis_results
        })
        
        return response.content
    except Exception as e:
        return f"Error reconstructing use case: {str(e)}"

# Data Analysis Tools
def get_available_tools(df):
    """Return available analysis tools based on data characteristics"""
    tools = []
    
    # Basic tools available for any dataset
    tools.append({
        "name": "data_profiler",
        "description": "Generate comprehensive data profile with statistics and quality assessment",
        "icon": "📊"
    })
    
    tools.append({
        "name": "data_cleaner",
        "description": "Identify and handle missing values, outliers, and inconsistencies",
        "icon": "🧹"
    })
    
    # Numeric data tools
    if len(df.select_dtypes(include=['number']).columns) > 0:
        tools.append({
            "name": "numeric_analysis",
            "description": "Perform advanced statistical analysis on numeric columns",
            "icon": "🔢"
        })
        
        if len(df.select_dtypes(include=['number']).columns) >= 2:
            tools.append({
                "name": "correlation_analysis",
                "description": "Analyze relationships between numeric variables",
                "icon": "🔄"
            })
    
    # Categorical data tools
    if len(df.select_dtypes(include=['object', 'category']).columns) > 0:
        tools.append({
            "name": "categorical_analysis",
            "description": "Analyze frequency distributions and patterns in categorical data",
            "icon": "📋"
        })
    
    # Time series tools
    if len(df.select_dtypes(include=['datetime']).columns) > 0:
        tools.append({
            "name": "time_series_analysis",
            "description": "Analyze trends, seasonality, and patterns over time",
            "icon": "⏳"
        })
    
    # Advanced tools
    tools.append({
        "name": "predictive_modeling",
        "description": "Build simple predictive models (regression/classification)",
        "icon": "🔮"
    })
    
    tools.append({
        "name": "segmentation_analysis",
        "description": "Identify natural groupings or segments in the data",
        "icon": "👥"
    })
    
    tools.append({
        "name": "anomaly_detection",
        "description": "Identify unusual patterns or outliers in the data",
        "icon": "⚠️"
    })
    
    return tools

def run_tool_analysis(tool_name, df, use_case=None):
    """Execute specific analysis tool"""
    try:
        if tool_name == "data_profiler":
            return {"result": get_data_summary(df)}
            
        elif tool_name == "numeric_analysis":
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) == 0:
                return {"error": "No numeric columns found"}
                
            analysis = "### Numeric Analysis\n"
            for col in numeric_cols:
                analysis += f"""
                **{col}**
                - Skewness: {df[col].skew():.2f}
                - Kurtosis: {df[col].kurtosis():.2f}
                - 25th percentile: {df[col].quantile(0.25):.2f}
                - 75th percentile: {df[col].quantile(0.75):.2f}
                - IQR: {df[col].quantile(0.75) - df[col].quantile(0.25):.2f}
                - Outliers: {len(df[(df[col] < (df[col].quantile(0.25) - 1.5*(df[col].quantile(0.75) - df[col].quantile(0.25)))) | 
                                  (df[col] > (df[col].quantile(0.75) + 1.5*(df[col].quantile(0.75) - df[col].quantile(0.25))))])}
                """
            return {"result": analysis}
            
        elif tool_name == "correlation_analysis":
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) < 2:
                return {"error": "Need at least 2 numeric columns for correlation analysis"}
                
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr, text_auto=True, title="Correlation Matrix")
            return {"result": "### Correlation Analysis", "visualization": fig}
            
        elif tool_name == "categorical_analysis":
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols) == 0:
                return {"error": "No categorical columns found"}
                
            analysis = "### Categorical Analysis\n"
            visuals = []
            for col in cat_cols:
                analysis += f"""
                **{col}**
                - Unique values: {df[col].nunique()}
                - Most frequent: {df[col].mode()[0]} ({(df[col] == df[col].mode()[0]).mean()*100:.1f}%)
                - Entropy: {-(df[col].value_counts(normalize=True) * np.log(df[col].value_counts(normalize=True))).sum():.2f}
                """
                
                # Add bar chart for top categories
                value_counts = df[col].value_counts().head(10)
                fig = px.bar(x=value_counts.index, y=value_counts.values,
                            title=f'Top Categories in {col}')
                visuals.append(fig)
                
            return {"result": analysis, "visualizations": visuals}
            
        elif tool_name == "time_series_analysis":
            datetime_cols = df.select_dtypes(include=['datetime']).columns
            if len(datetime_cols) == 0:
                return {"error": "No datetime columns found"}
                
            analysis = "### Time Series Analysis\n"
            visuals = []
            for dt_col in datetime_cols:
                analysis += f"""
                **{dt_col}**
                - Date range: {df[dt_col].min()} to {df[dt_col].max()}
                - Duration: {df[dt_col].max() - df[dt_col].min()}
                """
                
                # Add time series decomposition for numeric columns
                numeric_cols = df.select_dtypes(include=['number']).columns
                for num_col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                    try:
                        fig = px.line(df.sort_values(dt_col), x=dt_col, y=num_col,
                                    title=f'{num_col} over Time')
                        visuals.append(fig)
                    except:
                        pass
                        
            return {"result": analysis, "visualizations": visuals}
            
        else:
            return {"error": f"Tool {tool_name} not implemented yet"}
            
    except Exception as e:
        return {"error": f"Error in {tool_name}: {str(e)}"}

# Create agentic workflow
def create_agentic_workflow(llm, data_context, use_case=None):
    try:
        class AnalysisState(langchain.pydantic_v1.BaseModel):
            data_context: str
            use_case: str = ""
            analysis_results: str = None
            business_insights: str = None
            visualizations: List[Tuple[str, str, Any]] = []
            final_report: str = None
            conversation_history: List[Dict] = []
            reconstructed_use_case: str = None

        workflow = StateGraph(AnalysisState)

        def data_profiler(state):
            try:
                prompt = """You are a senior data analyst. Analyze this dataset:
                {data_context}
                
                User's use case: {use_case}
                
                Provide:
                1. Data quality assessment
                2. Key statistical findings
                3. Interesting patterns
                4. Potential analysis directions
                
                Be specific with numbers and clear explanations."""
                
                chain = PromptTemplate.from_template(prompt) | llm
                response = chain.invoke({
                    "data_context": state.data_context,
                    "use_case": state.use_case
                })
                
                return {"analysis_results": response.content}
            except Exception as e:
                return {"analysis_results": f"Error in data profiling: {str(e)}"}

        def visualization_specialist(state):
            try:
                if st.session_state.df is not None:
                    visualizations = visualize_data(st.session_state.df, state.use_case)
                    return {"visualizations": visualizations}
                return {"visualizations": []}
            except Exception as e:
                return {"visualizations": [], "analysis_results": f"{state.analysis_results}\nVisualization error: {str(e)}"}

        def business_consultant(state):
            try:
                prompt = """You are a business strategy consultant. Based on this analysis:
                {analysis_results}
                
                And the user's use case: {use_case}
                
                Generate:
                1. Key business insights
                2. Strategic recommendations
                3. Potential risks
                4. Implementation roadmap
                
                Make your response actionable and specific."""
                
                chain = PromptTemplate.from_template(prompt) | llm
                response = chain.invoke({
                    "analysis_results": state.analysis_results,
                    "use_case": state.use_case
                })
                
                return {"business_insights": response.content}
            except Exception as e:
                return {"business_insights": f"Error in business consulting: {str(e)}"}

        def use_case_reconstructor(state):
            try:
                if st.session_state.df is not None:
                    reconstructed = reconstruct_use_case(st.session_state.df, state.analysis_results)
                    return {"reconstructed_use_case": reconstructed}
                return {"reconstructed_use_case": "No data available for use case reconstruction"}
            except Exception as e:
                return {"reconstructed_use_case": f"Error in use case reconstruction: {str(e)}"}

        def report_generator(state):
            try:
                prompt = """Combine these elements into a professional report:
                
                DATA ANALYSIS:
                {analysis_results}
                
                BUSINESS INSIGHTS:
                {business_insights}
                
                USER USE CASE:
                {use_case}
                
                RECONSTRUCTED USE CASE:
                {reconstructed_use_case}
                
                Include:
                1. Executive summary
                2. Methodology
                3. Key findings
                4. Recommendations
                5. Next steps
                
                Use markdown formatting with headings, bullet points, and highlights."""
                
                chain = PromptTemplate.from_template(prompt) | llm
                response = chain.invoke({
                    "analysis_results": state.analysis_results,
                    "business_insights": state.business_insights,
                    "use_case": state.use_case,
                    "reconstructed_use_case": state.reconstructed_use_case
                })
                
                return {"final_report": response.content}
            except Exception as e:
                return {"final_report": f"Error generating report: {str(e)}"}

        def conversation_handler(state):
            try:
                if state.conversation_history:
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", """You are a helpful data analysis assistant. 
                        Continue the conversation based on the analysis context:
                        
                        Data Context: {data_context}
                        Current Analysis: {analysis_results}
                        Business Insights: {business_insights}
                        Reconstructed Use Case: {reconstructed_use_case}
                        
                        Be helpful, concise, and professional."""),
                        MessagesPlaceholder(variable_name="conversation_history"),
                        ("human", "{latest_query}"),
                    ])
                    
                    chain = prompt | llm
                    response = chain.invoke({
                        "data_context": state.data_context,
                        "analysis_results": state.analysis_results,
                        "business_insights": state.business_insights,
                        "reconstructed_use_case": state.reconstructed_use_case,
                        "conversation_history": state.conversation_history,
                        "latest_query": state.conversation_history[-1]["human"]
                    })
                    
                    return {"conversation_history": state.conversation_history + [{"ai": response.content}]}
                return {}
            except Exception as e:
                return {"conversation_history": state.conversation_history + [{"ai": f"Conversation error: {str(e)}"}]}

        # Add nodes
        workflow.add_node("profiler", data_profiler)
        workflow.add_node("visualizer", visualization_specialist)
        workflow.add_node("consultant", business_consultant)
        workflow.add_node("use_case_reconstructor", use_case_reconstructor)
        workflow.add_node("reporter", report_generator)
        workflow.add_node("conversation", conversation_handler)

        # Define edges
        workflow.add_edge("profiler", "visualizer")
        workflow.add_edge("visualizer", "consultant")
        workflow.add_edge("consultant", "use_case_reconstructor")
        workflow.add_edge("use_case_reconstructor", "reporter")
        workflow.add_edge("reporter", END)
        workflow.add_conditional_edges(
            "conversation",
            lambda state: END if not state.conversation_history else "profiler",
        )

        workflow.set_entry_point("profiler")
        return workflow.compile()
    except Exception as e:
        st.error(f"Error creating workflow: {str(e)}")
        return None

# Run analysis
def run_analysis(df, use_case=None):
    try:
        llm = initialize_llm()
        if not llm:
            return {"error": "Failed to initialize LLM"}
        
        data_context = get_data_summary(df)
        workflow = create_agentic_workflow(llm, data_context, use_case)
        if not workflow:
            return {"error": "Failed to create workflow"}

        results = workflow.invoke({
            "data_context": data_context,
            "use_case": use_case or "",
            "analysis_results": None,
            "business_insights": None,
            "visualizations": [],
            "final_report": None,
            "conversation_history": [],
            "reconstructed_use_case": None
        })
        
        return {
            "data_context": data_context,
            "analysis": results.get("analysis_results", ""),
            "insights": results.get("business_insights", ""),
            "report": results.get("final_report", ""),
            "visualizations": results.get("visualizations", []),
            "reconstructed_use_case": results.get("reconstructed_use_case", "")
        }
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        traceback.print_exc()
        return {"error": f"Analysis error: {str(e)}"}

# Display conversation
def display_conversation():
    try:
        for message in st.session_state.conversation:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(f'<div class="user-query">{message["content"]}</div>', unsafe_allow_html=True)
            else:
                with st.chat_message("assistant"):
                    st.markdown(f'<div class="ai-response">{message["content"]}</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error displaying conversation: {str(e)}")

# Display available tools
def display_tool_selection(df):
    st.subheader("🔧 Analysis Tools")
    st.markdown("Select the tools you want to apply to your data:")
    
    available_tools = get_available_tools(df)
    selected_tools = []
    
    cols = st.columns(3)
    for i, tool in enumerate(available_tools):
        with cols[i % 3]:
            if st.checkbox(f"{tool['icon']} {tool['name']}", key=f"tool_{tool['name']}"):
                selected_tools.append(tool['name'])
    
    if st.button("Run Selected Tools"):
        if not selected_tools:
            st.warning("Please select at least one tool")
        else:
            with st.spinner("Running selected tools..."):
                for tool in selected_tools:
                    with st.expander(f"{tool.replace('_', ' ').title()}"):
                        result = run_tool_analysis(tool, df)
                        if "error" in result:
                            st.error(result["error"])
                        else:
                            if "result" in result:
                                st.markdown(result["result"])
                            if "visualization" in result:
                                st.plotly_chart(result["visualization"], use_container_width=True)
                            if "visualizations" in result:
                                for viz in result["visualizations"]:
                                    st.plotly_chart(viz, use_container_width=True)

# Main app logic
def main():
    init_session_state()
    use_case = render_sidebar()

    if st.session_state.uploaded_file is not None and st.session_state.df is not None:
        try:
            st.subheader("📂 Uploaded Dataset")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**File Name:** {st.session_state.uploaded_file.name}")
                st.write(f"**Shape:** {st.session_state.df.shape[0]} rows, {st.session_state.df.shape[1]} columns")
            
            with col2:
                st.write("**First 5 Rows:**")
                st.dataframe(st.session_state.df.head())
            
            st.markdown("---")
            
            # Display tool selection interface
            display_tool_selection(st.session_state.df)
            
            if use_case and st.button("🚀 Run Full Analysis"):
                with st.spinner("Running analysis with AI agents..."):
                    results = run_analysis(st.session_state.df, use_case)
                    st.session_state.analysis_results = results
                    
                    if "error" not in results:
                        st.subheader("📈 Analysis Results")
                        st.markdown(results["analysis"])
                        
                        st.subheader("💡 Business Insights")
                        st.markdown(results["insights"])
                        
                        st.subheader("🔍 Reconstructed Use Case")
                        st.markdown(results["reconstructed_use_case"])
                        
                        st.subheader("📝 Full Report")
                        st.markdown(results["report"])
                        
                        if results["visualizations"]:
                            st.subheader("📊 Visualizations")
                            for viz_type, viz_name, fig in results["visualizations"]:
                                st.write(f"**{viz_type.capitalize()}: {viz_name}**")
                                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            display_conversation()
        
        except Exception as e:
            st.error(f"Main display error: {str(e)}")
            traceback.print_exc()
    
    else:
        st.info("Please upload a CSV file to begin analysis.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        traceback.print_exc()