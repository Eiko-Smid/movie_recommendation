import streamlit as st

st.set_page_config(page_title="Movie Recommender – Future Improvements", page_icon="🎬", layout="centered")
st.title("🎯 Future Improvements & Outlook")

st.markdown("---")


# Define improvements (now with bullet-list details)
improvements = [
    {
        "title": "⚙️ Pipeline & Automation",
        "summary": "Move from a simple cron-based retraining setup to an orchestrated ML pipeline.",
        "description": """
        Transition to an <b>Apache Airflow DAG</b> for robust orchestration:<br />
        - Schedule model retraining and evaluation workflows<br />
        - Manage task dependencies and retries<br />
        - Enable dynamic workflows<br />
        """
    },
    {
        "title": "📊 Visualization",
        "summary": "Introduce centralized dashboards for observability and metric tracking.",
        "description": """
        Combine <b>Prometheus</b> (for metric scraping) with <b>Grafana</b> (for visualization) to:<br />
        - Track API latency, errors, and throughput<br />
        - Monitor model KPIs such as Precision@K and Recall<br />
        - Visualize retraining frequency and outcomes<br />
        """
    },
    {
        "title": "🧠 Monitoring",
        "summary": "Continuously assess data and model health to prevent performance drift.",
        "description": """
        Integrate <b>Evidently AI</b> for data and model monitoring:<br />
        - Detect data drift and concept drift automatically<br />
        - Compare live vs training data distributions<br />
        - Trigger retraining or alert notifications<br />
        """
    },
    {
        "title": "🔒 API Security",
        "summary": "Ensure secure access to API endpoints and model services.",
        "description": """
        Add strong <b>authentication and authorization</b>:<br />
        - Use OAuth2 or JWT tokens for secure access<br />
        - Implement Role-Based Access Control (RBAC)<br />
        - Log access for audit and compliance<br />
        """
    },
    {
        "title": "📈 Scalability",
        "summary": "Prepare the system for real-world traffic and dynamic workloads.",
        "description": """
        Deploy in a <b>containerized and auto-scaled environment</b>:<br />
        - Use Kubernetes or AWS ECS for deployment<br />
        - Enable horizontal auto-scaling during traffic spikes<br />
        - Maintain low latency and cost efficiency<br />
        """
    }
]

# Display improvements as visually structured cards
for item in improvements:
    st.markdown(f"""
    <div style='background-color:#f8f9fa; padding: 1.4rem; border-radius: 12px;
                margin-bottom: 1.2rem; border-left: 6px solid #00BFFF;
                box-shadow: 0 2px 6px rgba(0,0,0,0.06);'>
        <h4 style='margin-bottom: 0.4rem;'>{item["title"]}</h4>
        <p style='margin: 0.2rem 0 0.6rem 0; color:#555; font-style: italic;'>{item["summary"]}</p>
        <hr style='border: none; border-top: 1px solid #eee; margin: 0.6rem 0;' />
        <div style='line-height: 1.6;'>{item["description"]}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.info("💡 These steps will help move the recommender system from a prototype toward a production-grade MLOps pipeline.")