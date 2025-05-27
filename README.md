# Data Analysis AI Assistant

## Overview
The **Data Analysis AI Assistant** is a powerful, AI-driven web application designed to streamline and enhance data analysis workflows. Built with **Streamlit**, **LangGraph**, and **Grok**, this tool allows users to upload CSV files, perform automated data profiling, generate interactive visualizations, and derive actionable business insights through an agentic workflow. It is ideal for data analysts, business professionals, and researchers seeking efficient and intelligent data exploration.

## Features
- **Automated Data Profiling**: Provides comprehensive summaries, including dataset statistics, column details, and correlation analysis.
- **Interactive Visualizations**: Generates dynamic charts (histograms, box plots, scatter plots, heatmaps, etc.) using Plotly.
- **Agentic Workflow**: Utilizes LangGraph and Grok for multi-step analysis, including data profiling, visualization, business insights, and use case reconstruction.
- **Customizable Analysis Tools**: Offers tools like numeric analysis, correlation analysis, and time series analysis, tailored to the dataset's characteristics.
- **Conversational Interface**: Supports interactive user queries to refine analysis and explore data further.
- **Professional Reporting**: Combines analysis results, visualizations, and insights into a structured markdown report.

## Tech Stack
- **Python**: Core programming language
- **Streamlit**: Web application framework for the user interface
- **LangGraph**: Orchestrates agentic workflows for multi-step analysis
- **Grok (via LangChain)**: AI model for intelligent data interpretation and insights
- **Pandas & NumPy**: Data manipulation and analysis
- **Plotly**: Interactive visualizations
- **Seaborn & Matplotlib**: Additional visualization support

## Installation

### Prerequisites
- Python 3.8+
- A Groq API key (set as an environment variable: `GROQ_API_KEY`)
- A compatible web browser (e.g., Chrome, Firefox)

### Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Mr-Amresh/DataAnayzerLangraph.git
   cd DataAnayzerLangraph
   ```

2. **Install Dependencies**:
   Create a virtual environment and install the required packages:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set Environment Variables**:
   Export your Groq API key:
   ```bash
   export GROQ_API_KEY="your-groq-api-key"
   ```

4. **Run the Application**:
   Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

5. **Access the App**:
   Open your browser and navigate to `http://localhost:8501`.

### Requirements
Install the required Python packages using:
```bash
pip install streamlit pandas numpy plotly langchain langgraph langchain-groq matplotlib seaborn
```

## Usage
1. **Upload a CSV File**: Use the sidebar to upload a CSV file for analysis.
2. **Specify Analysis Goals**: Enter your use case or analysis objectives in the provided text area.
3. **Select Analysis Tools**: Choose from available tools (e.g., data profiler, correlation analysis) to customize your analysis.
4. **Run Full Analysis**: Click the "Run Full Analysis" button to generate a comprehensive report, including data summaries, visualizations, and business insights.
5. **Explore Visualizations**: View interactive charts to understand data distributions, correlations, and trends.
6. **Review Reports**: Access the generated report with executive summaries, methodologies, and recommendations.

## Example Workflow
1. Upload a CSV file containing sales data.
2. Specify a use case, e.g., "Analyze sales trends and identify top-performing products."
3. Select tools like "data_profiler" and "time_series_analysis."
4. Run the analysis to receive:
   - A data summary with key statistics
   - Visualizations (e.g., line charts for sales over time)
   - Business insights (e.g., top products, seasonal trends)
   - A reconstructed use case and final report

## Limitations
- Only CSV files are supported for data input.
- Large datasets (>500MB) may impact performance.
- Some advanced tools (e.g., predictive modeling) are in beta and may require additional configuration.
- Requires a stable internet connection for Groq API access.

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository: `https://github.com/Mr-Amresh/DataAnayzerLangraph`
2. Create a feature branch (`git checkout -b feature/new-feature`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature/new-feature`).
5. Open a pull request.

Please ensure your code follows PEP 8 guidelines and includes appropriate tests.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For questions or support, please contact [your-email@example.com](mailto:your-email@example.com) or open an issue on the GitHub repository: [https://github.com/Mr-Amresh/DataAnayzerLangraph](https://github.com/Mr-Amresh/DataAnayzerLangraph).

## Acknowledgments
- Built with ❤️ using [Streamlit](https://streamlit.io/), [LangGraph](https://github.com/langchain-ai/langgraph), and [Grok](https://x.ai/grok) by xAI.
- Special thanks to the open-source community for their invaluable contributions to the data science ecosystem.
