import io
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Enable interactive mode for matplotlib
plt.ion()

st.set_page_config(page_title="CSV Data Plotter", page_icon="📊", layout="wide")

# Create two main columns for the layout
left_col, right_col = st.columns([1, 1], gap="large")

# Initialize df as None
df = None

with left_col:
    st.title("📊 CSV Data Plotter")
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write("### Data Preview")
        st.dataframe(df.head(50), use_container_width=True)

# Only proceed with the rest of the app if we have data
if df is not None:

    # Move chart controls and preview to right column
    with right_col:
        st.header("Chart Preview")
        
        # Chart controls in a container
        with st.container():
            st.subheader("Chart Controls")
            if df is None:
                st.info("Please upload a CSV file to begin")
            
            cols = list(df.columns)
            x_col = st.selectbox("X-axis column", cols, index=0)
            
            # Get all numerical columns
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            numeric_cols.append("Count")  # Add Count as an option
            
            # Y-axis selection
            y_col = st.selectbox("Y-axis column", numeric_cols, index=len(numeric_cols)-1)  # Set Count as default

            # Chart configuration in two rows
            control_col1, control_col2 = st.columns(2)
            with control_col1:
                chart_type = st.selectbox("Chart type", ["Bar", "Horizontal Bar", "Line", "Area", "Pie"], index=0)
                max_items = st.select_slider(
                    "Maximum items to show",
                    options=[5, 10, 15, 20, 25, 30, 'All'],
                    value=15
                )
            with control_col2:
                sort_options = ["Value (Descending)", "Value (Ascending)", "Name (A-Z)", "Name (Z-A)"] if y_col != "Count" else ["Count (Descending)", "Count (Ascending)", "Name (A-Z)", "Name (Z-A)"]
                sort_by = st.selectbox("Sort by", sort_options, index=0)

    # Prepare data
    if y_col == "Count":
        # For Count, group by X and count occurrences
        plot_df = df.groupby(x_col).size().reset_index(name="Count")
    else:
        # For numeric columns, calculate mean, min, max, and count
        agg_dict = {
            y_col: ['mean', 'min', 'max', 'count']
        }
        plot_df = df.groupby(x_col).agg(agg_dict).reset_index()
        plot_df.columns = [x_col, f"{y_col}_mean", f"{y_col}_min", f"{y_col}_max", f"{y_col}_count"]
        
        # Add statistics to the left column
        with left_col:
            st.write(f"### Statistics for {y_col}")
            stats_cols = st.columns(4)
            with stats_cols[0]:
                st.metric("Mean", f"{plot_df[f'{y_col}_mean'].mean():.2f}")
            with stats_cols[1]:
                st.metric("Min", f"{plot_df[f'{y_col}_min'].min():.2f}")
            with stats_cols[2]:
                st.metric("Max", f"{plot_df[f'{y_col}_max'].max():.2f}")
            with stats_cols[3]:
                st.metric("Total Count", f"{plot_df[f'{y_col}_count'].sum():,}")
        
        # Use mean for plotting
        plot_df = plot_df[[x_col, f"{y_col}_mean"]].rename(columns={f"{y_col}_mean": y_col})
    
    # Sort data based on user selection
    if sort_by == "Count (Descending)" or sort_by == "Value (Descending)":
        plot_df = plot_df.sort_values(y_col, ascending=False)
    elif sort_by == "Count (Ascending)" or sort_by == "Value (Ascending)":
        plot_df = plot_df.sort_values(y_col, ascending=True)
    elif sort_by == "Name (A-Z)":
        plot_df = plot_df.sort_values(x_col)
    else:  # Name (Z-A)
        plot_df = plot_df.sort_values(x_col, ascending=False)
    
    # Limit the number of items if not 'All'
    if max_items != 'All':
        plot_df = plot_df.head(max_items)

    # Create the chart with a larger figure size for better interactivity
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.canvas.toolbar_visible = True
    fig.canvas.header_visible = True
    
    # Enable grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
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
            ax.set_xlabel(y_col)
        else:
            ax.set_ylabel(y_col)
    
    ax.set_title(f"{chart_type} Chart: {y_col} by {x_col}")
    
    # Adjust label size and rotation
    if chart_type != "Pie":
        if chart_type == "Horizontal Bar":
            plt.yticks(fontsize=8)
        else:
            plt.xticks(fontsize=8, rotation=45, ha='right')
            plt.subplots_adjust(bottom=0.2)
    
    # Automatically adjust subplot parameters for better layout
    plt.tight_layout()

    # Create chart container in the right column
    with right_col:
        chart_container = st.container()
        with chart_container:
            st.markdown(
                """
                <style>
                    .stImage > img {
                        max-width: 100% !important;
                        max-height: 320px !important;
                    }
                </style>
                """, 
                unsafe_allow_html=True
            )
            st.pyplot(fig, use_container_width=True)

        # ✅ Download chart as PNG - moved to left column
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        
    # Add download button to left column
    with left_col:
        st.download_button(
            "Download chart as PNG",
            buf.getvalue(),
            file_name="chart.png",
            mime="image/png"
        )
