import io
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="CSV Data Plotter", page_icon="📊", layout="wide")
st.title("📊 CSV Data Plotter")

uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.write("### Data Preview")
    st.dataframe(df.head(50), use_container_width=True)

    cols = list(df.columns)
    x_col = st.selectbox("X-axis column", cols, index=0)
    y_col = "Count"  # Fixed to Count

    # Chart configuration
    col1, col2, col3 = st.columns(3)
    with col1:
        chart_type = st.selectbox("Chart type", ["Bar", "Horizontal Bar", "Line", "Area", "Pie"], index=0)
    with col2:
        max_items = st.select_slider(
            "Maximum items to show",
            options=[5, 10, 15, 20, 25, 30, 'All'],
            value=15
        )
    with col3:
        sort_by = st.selectbox("Sort by", ["Count (Descending)", "Count (Ascending)", "Name (A-Z)", "Name (Z-A)"], index=0)

    # Prepare data
    plot_df = df.groupby(x_col).size().reset_index(name="Count")
    
    # Sort data based on user selection
    if sort_by == "Count (Descending)":
        plot_df = plot_df.sort_values("Count", ascending=False)
    elif sort_by == "Count (Ascending)":
        plot_df = plot_df.sort_values("Count", ascending=True)
    elif sort_by == "Name (A-Z)":
        plot_df = plot_df.sort_values(x_col)
    else:  # Name (Z-A)
        plot_df = plot_df.sort_values(x_col, ascending=False)
    
    # Limit the number of items if not 'All'
    if max_items != 'All':
        plot_df = plot_df.head(max_items)

    # Create the chart
    fig, ax = plt.subplots()

    # Convert x-axis values to strings for categorical plotting
    plot_df[x_col] = plot_df[x_col].astype(str)

    if chart_type == "Bar":
        ax.bar(plot_df[x_col], plot_df[y_col])
    elif chart_type == "Horizontal Bar":
        ax.barh(plot_df[x_col], plot_df[y_col])
    elif chart_type == "Line":
        ax.plot(plot_df[x_col], plot_df[y_col], marker='o')
    elif chart_type == "Area":
        ax.fill_between(range(len(plot_df)), plot_df[y_col], alpha=0.5)
        ax.plot(range(len(plot_df)), plot_df[y_col], marker='o')
        ax.set_xticks(range(len(plot_df)))
        ax.set_xticklabels(plot_df[x_col])
    elif chart_type == "Pie":
        ax.pie(plot_df[y_col], labels=plot_df[x_col].astype(str), autopct="%1.1f%%")
    
    # Set labels and title
    if chart_type != "Horizontal Bar":
        ax.set_xlabel(x_col)
    if chart_type != "Pie":
        if chart_type == "Horizontal Bar":
            ax.set_ylabel(x_col)
            ax.set_xlabel("Count")
        else:
            ax.set_ylabel("Count")
    
    ax.set_title(f"{chart_type} Chart: Count by {x_col}")
    
    # Adjust label size and rotation
    if chart_type != "Pie":
        if chart_type == "Horizontal Bar":
            plt.yticks(fontsize=8)
        else:
            plt.xticks(fontsize=8, rotation=45, ha='right')
            plt.subplots_adjust(bottom=0.2)
    
    # Automatically adjust subplot parameters for better layout
    plt.tight_layout()

    # ✅ Render at fixed size (not stretched)
    # Create a container with fixed width
    chart_container = st.container()
    with chart_container:
        st.markdown(
            """
            <style>
                .stImage > img {
                    max-width: 1024px !important;
                    max-height: 512px !important;
                }
            </style>
            """, 
            unsafe_allow_html=True
        )
        st.pyplot(fig, use_container_width=False)

    # ✅ Download chart as PNG
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    st.download_button(
        "Download chart as PNG",
        buf.getvalue(),
        file_name="chart.png",
        mime="image/png"
    )
