import streamlit as st

st.set_page_config(page_title="Movie Recommender – Future Improvements", page_icon="🎬", layout="centered")
st.title("🎯 Future Improvements & Outlook")

st.markdown("---")

# Define improvements
improvements = [
    {
        "title": "⚙️ Pipeline & Automation",
        "description": "Upgrade the simple cron-based retraining to an <b>Airflow DAG</b> for better scheduling, dependency management, and monitoring."
    },
    {
        "title": "📊 Visualization",
        "description": "Integrate <b>Grafana + Prometheus</b> dashboards to track system metrics, model KPIs, and retraining performance."
    },
    {
        "title": "🧠 Monitoring",
        "description": "Use <b>Evidently AI</b> to detect data drift and model degradation early. Trigger automated retraining or alert notifications."
    },
    {
        "title": "🔒 API Security",
        "description": "Add <b>authentication</b> (e.g., OAuth2 / JWT) to restrict access to registered or internal users only."
    },
    {
        "title": "📈 Scalability",
        "description": "Enable <b>auto-scaling</b> with Kubernetes or ECS to handle production traffic spikes while maintaining availability."
    }
]

# Display in styled boxes
for item in improvements:
    st.markdown(f"""
    <div style='background-color:#f8f9fa; padding: 1rem; border-radius: 10px; margin-bottom: 0.8rem; border-left: 5px solid #00BFFF;'>
        <h4 style='margin-bottom: 0.3rem;'>{item["title"]}</h4>
        <p style='margin: 0;'>{item["description"]}</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.info("💡 These steps will help move the recommender system from a prototype toward a production-grade MLOps pipeline.")
