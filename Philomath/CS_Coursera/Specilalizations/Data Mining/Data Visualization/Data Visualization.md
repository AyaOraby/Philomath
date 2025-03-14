# Data Visualization Course

## Week 1: Introduction to Data Visualization

### Understanding Data Visualization
Data visualization is the graphical representation of information and data. By using visual elements like charts, graphs, and maps, data visualization tools make data more accessible and easier to understand. This approach helps in identifying patterns, trends, and outliers in large datasets.

Effective data visualization is crucial for decision-making in various fields, including business, science, and technology. It allows analysts to convey complex information in a way that is intuitive and easy to interpret.

### 2D and 3D Graphics in Visualization

#### **2D Graphics**
2D visualizations involve flat representations of data on a two-dimensional plane. They are widely used due to their simplicity and ease of interpretation. Some common 2D visualization types include:

- **Bar Charts:** Ideal for comparing discrete categories.
- **Line Graphs:** Show trends over time.
- **Scatter Plots:** Represent relationships between variables.
- **Histograms:** Depict frequency distributions.

##### **SVG (Scalable Vector Graphics)**
SVG is a widely used XML-based format for rendering resolution-independent 2D vector graphics on the web. It allows precise control over shapes, colors, and text rendering.

##### **Matplotlib (Python) & D3.js (JavaScript)**
- **Matplotlib:** A powerful Python library for creating static, animated, and interactive plots. It provides extensive customization options.
- **D3.js:** A JavaScript library that enables dynamic and interactive data visualizations using web technologies like HTML, CSS, and SVG.

##### **Creating a Bar Chart in Python**
```python
import matplotlib.pyplot as plt

categories = ['A', 'B', 'C', 'D']
values = [10, 20, 15, 25]

plt.bar(categories, values, color='blue')
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Bar Chart Example')
plt.show()
```

##### **Creating a Line Chart in Python**
```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Simple Line Chart')
plt.show()
```

#### **3D Graphics**
3D visualizations add depth, making them suitable for complex datasets, simulations, and engineering models.

- **Photorealistic Rendering:** Mimics real-world appearance using shading, lighting, and textures.
- **Non-Photorealistic Rendering:** Focuses on stylization and abstraction, highlighting essential data features.

##### **Creating a 3D Scatter Plot in Python**
```python
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = np.random.rand(50)
y = np.random.rand(50)
z = np.random.rand(50)

ax.scatter(x, y, z, c='r', marker='o')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
plt.title('3D Scatter Plot')
plt.show()
```

### Human Perception and Visualization
Understanding human perception is crucial for designing effective visualizations. Some key principles include:

- **The Role of the Retina:** The human eye processes information by detecting colors, shapes, and depth. Effective visualizations leverage these factors to enhance clarity.
- **Memory and Cognitive Processing:** Well-designed visualizations use familiar patterns, logical grouping, and color coding to improve comprehension and retention.
- **Perceiving Dimensions:** Techniques such as shading and vanishing points simulate three-dimensionality, improving depth perception.

---

## Week 2: Data Mapping and Charting

### Data Representation Techniques
Different types of data require specific visualization methods.

- **Numerical Data:** Continuous and discrete numerical values are best represented using scatter plots, histograms, and line graphs.
- **Categorical Data:** Bar charts and pie charts are useful for showing distributions.
- **Temporal Data:** Time-series graphs such as line charts and area charts effectively show trends over time.

#### **Data Transformation**
Raw data often needs preprocessing to be effectively visualized:
- **Scaling:** Normalizing values to fit within a specific range.
- **Binning:** Grouping numerical data into discrete intervals.
- **Aggregation:** Summarizing data points for better clarity.

#### **Chart Types**
- **Bar Charts:** Suitable for comparing categorical data.
- **Line Charts:** Used for displaying trends over time.
- **Scatter Plots:** Show relationships between numerical variables.

### Advanced Visualization Techniques
- **Glyphs:** Symbols representing individual data points for multi-dimensional analysis.
- **Parallel Coordinates:** Used to visualize multi-variable data by plotting each variable as a parallel axis.
- **Stacked Graphs:** Display cumulative trends over time.

### Design Principles
- **Tufte's Guidelines:** Emphasize minimal clutter, a high data-to-ink ratio, and clear communication.
- **Color Usage:** Proper color selection enhances readability, differentiates categories, and improves accessibility.

### Scatter Plot Example in Python
```python
import matplotlib.pyplot as plt
import numpy as np

x = np.random.rand(50)
y = np.random.rand(50)

plt.scatter(x, y, color='blue')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot Example')
plt.show()
```



### Histogram Example in Python
```python
import matplotlib.pyplot as plt
import numpy as np

data = np.random.randn(1000)
plt.hist(data, bins=30, color='blue', alpha=0.7)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram Example')
plt.show()
```

---

## Week 3: Graph Visualization and Dimensionality Reduction

### Understanding Graphs and Networks
- **Graphs:** Represent relationships between entities using nodes and edges.
- **Embedding Planar Graphs:** Techniques to structure graph layouts and avoid clutter.
- **Tree Maps:** Display hierarchical data using nested rectangles.

### Dimensionality Reduction Techniques
- **Principal Component Analysis (PCA):** Reduces dimensions while preserving essential information.
- **Multidimensional Scaling (MDS):** Visualizes high-dimensional data in a lower-dimensional space.
- **Packing Algorithms:** Used for arranging elements efficiently in a constrained space.

### Creating a Network Graph with NetworkX
```python
import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1), (1, 3)])
nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray')
plt.title('Network Graph')
plt.show()
```

### PCA Visualization
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Visualization')
plt.colorbar()
plt.show()
```

---

## Week 4: Information Visualization and System Design

### Creating an Interactive Dashboard with Dash (Plotly)
```python
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd

app = dash.Dash(__name__)

df = px.data.iris()
fig = px.scatter(df, x='sepal_width', y='sepal_length', color='species')

app.layout = html.Div([
    dcc.Graph(id='scatter-plot', figure=fig)
])

if __name__ == '__main__':
    app.run_server(debug=True)
```

