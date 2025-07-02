from bs4 import BeautifulSoup
import os

def extract_elements_from_html(file_path, output_dir):
    with open(file_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Extract AQI Y-axis values
    y_axis = soup.find_all("g", class_="highcharts-axis-labels highcharts-yaxis-labels")
    with open(os.path.join(output_dir, "y_axis_labels.txt"), "w", encoding='utf-8') as f:
        for tag in y_axis:
            f.write(tag.get_text(separator="\n") + "\n")

    # Extract X-axis (Date/Time)
    x_axis = soup.find_all("g", class_="highcharts-axis-labels highcharts-xaxis-labels")
    with open(os.path.join(output_dir, "x_axis_labels.txt"), "w", encoding='utf-8') as f:
        for tag in x_axis:
            f.write(tag.get_text(separator="\n") + "\n")

    # Extract Series Data (AQI curve path)
    series_group = soup.find_all("g", class_="highcharts-series-group")
    with open(os.path.join(output_dir, "series_paths.txt"), "w", encoding='utf-8') as f:
        for group in series_group:
            paths = group.find_all("path")
            for path in paths:
                d_attr = path.get("d")
                if d_attr:
                    f.write(d_attr + "\n")

    print(f"✅ Extracted and saved elements from {os.path.basename(file_path)} to {output_dir}")

# Example usage:
extract_elements_from_html(
    "DelhiNCRAQI(1st July).txt",
    "./output_PM10"
)

extract_elements_from_html(
    "DelhiNCRAQI(PM 2.5).txt",
    "./output_PM25"
)
