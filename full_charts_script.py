
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list
from sklearn.preprocessing import StandardScaler
import networkx as nx
from pandas.plotting import parallel_coordinates, radviz

# Generate sample data
N = 5000
df = pd.DataFrame({
    'temperature': np.random.normal(5800, 2000, N),
    'luminosity': np.random.lognormal(0, 1, N),
    'radius': np.random.lognormal(0, 1, N),
    'absolute_magnitude': np.random.normal(5, 2, N),
    'distance': np.random.exponential(1000, N)
})
time = np.linspace(0, 10, N)

# 1. Line plot
plt.figure()
plt.plot(time, df['temperature'])
plt.title('Line: Temperature over time')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.show()

# 2. Bar chart
bins = pd.cut(df['absolute_magnitude'], bins=5)
counts = bins.value_counts().sort_index()
plt.figure()
plt.bar(range(len(counts)), counts)
plt.xticks(range(len(counts)), counts.index.astype(str), rotation=45)
plt.title('Bar: Absolute Magnitude buckets')
plt.show()

# 3. Mean luminosity per bucket
mean_lum = df.groupby(bins)['luminosity'].mean()
plt.figure()
plt.bar(range(len(mean_lum)), mean_lum)
plt.xticks(range(len(mean_lum)), mean_lum.index.astype(str), rotation=45)
plt.title('Mean Luminosity by bucket')
plt.ylabel('Mean Luminosity')
plt.show()

# 4. Donut chart
sizes = counts.values
labels = counts.index.astype(str)
plt.figure()
patches, texts, autotexts = plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
plt.gca().add_artist(plt.Circle((0,0), 0.70, color='white'))
plt.title('Donut: Magnitude buckets')
plt.show()

# 5. Bubble chart
plt.figure()
plt.scatter(df['temperature'], df['luminosity'], s=df['radius']*5, alpha=0.4)
plt.title('Bubble: Temp vs Lum vs Radius')
plt.xlabel('Temperature')
plt.ylabel('Luminosity')
plt.show()

# 6. Contour & Filled Contour
x, y = df['temperature'], df['luminosity']
xi = np.linspace(x.min(), x.max(), 100)
yi = np.linspace(y.min(), y.max(), 100)
zi, xedges, yedges = np.histogram2d(x, y, bins=[xi, yi])
X, Y = np.meshgrid((xedges[:-1]+xedges[1:])/2, (yedges[:-1]+yedges[1:])/2)
plt.figure()
plt.contour(X, Y, zi.T)
plt.title('Contour: Temp vs Lum')
plt.show()
plt.figure()
plt.contourf(X, Y, zi.T)
plt.title('Filled Contour')
plt.show()

# 7. 3D Surface plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(x, y, df['radius'], linewidth=0.2, antialiased=True)
ax.set_title('3D Surface: Radius')
plt.show()

# 8. Streamplot & Quiver
U = np.cos(X)
V = np.sin(Y)
plt.figure()
plt.streamplot(X, Y, U.T, V.T)
plt.title('Streamplot')
plt.show()
plt.figure()
plt.quiver(X, Y, U.T, V.T)
plt.title('Quiver')
plt.show()

# 9. Parallel Coordinates & RadViz
df_sample = df.sample(200).assign(
    category=pd.cut(df['absolute_magnitude'], 3, labels=['Low','Mid','High'])
)
plt.figure(figsize=(8,4))
parallel_coordinates(df_sample[['temperature','luminosity','radius','category']], 'category')
plt.title('Parallel Coordinates')
plt.show()
plt.figure()
radviz(df_sample, 'category')
plt.title('RadViz')
plt.show()

# 10. Dendrogram
scaled = StandardScaler().fit_transform(df_sample[['temperature','luminosity']])
link = linkage(scaled, method='ward')
plt.figure()
dendrogram(link)
plt.title('Dendrogram')
plt.show()

# 11. Ridgeline (stacked KDE approximation)
plt.figure()
for i, col in enumerate(['temperature','luminosity','radius']):
    data = df[col]
    hist, bins = np.histogram(data, bins=50, density=True)
    centers = (bins[:-1] + bins[1:]) / 2
    plt.fill_between(centers, i + hist, i, alpha=0.6)
plt.yticks([0,1,2], ['temperature','luminosity','radius'])
plt.title('Ridgeline KDE approx')
plt.show()

# 12. Clustered Heatmap (manual)
corr = df_sample[['temperature','luminosity','radius','distance']].corr()
link_corr = linkage(corr, method='average')
order = leaves_list(link_corr)
ordered = corr.iloc[order, order]
plt.figure()
plt.imshow(ordered, aspect='auto')
plt.colorbar()
plt.xticks(range(len(order)), ordered.columns, rotation=90)
plt.yticks(range(len(order)), ordered.index)
plt.title('Clustered Heatmap')
plt.show()

# 13. Network graph
G = nx.erdos_renyi_graph(15, 0.3)
plt.figure()
nx.draw(G, with_labels=True)
plt.title('Network Graph')
plt.show()

# 14. Sankey diagram
from matplotlib.sankey import Sankey
plt.figure()
Sankey().add(flows=[1, -0.6, -0.4], labels=['Total','Low mag','High mag'], orientations=[0,1,-1]).finish()
plt.title('Sankey Diagram')
plt.show()

# 15. Bullet chart
plt.figure()
plt.barh([0], [mean_lum.mean()])
plt.barh([0], [mean_lum.max()])
plt.yticks([])
plt.title('Bullet Chart')
plt.show()

# 16. Waterfall chart
diffs = np.diff(counts.values, prepend=0)
colors = ['green' if d >= 0 else 'red' for d in diffs]
plt.figure()
plt.bar(range(len(diffs)), diffs, color=colors)
plt.title('Waterfall Chart')
plt.show()

# 17. Area Chart (cumulative distribution)
dist_sorted = np.sort(df['distance'])
cum = np.arange(1, N+1) / N
plt.figure()
plt.fill_between(dist_sorted, cum)
plt.title('Cumulative Distribution')
plt.show()

# 18. Animation (scatter evolving)
fig, ax = plt.subplots()
scat = ax.scatter([], [], s=10)
def init():
    ax.set_xlim(df['temperature'].min(), df['temperature'].max())
    ax.set_ylim(df['luminosity'].min(), df['luminosity'].max())
    return scat,

def update(i):
    sample = df.sample(300)
    scat.set_offsets(np.c_[sample['temperature'], sample['luminosity']])
    return scat,

anim = animation.FuncAnimation(fig, update, init_func=init, frames=10, interval=300)
# To view animation in Jupyter, display: anim
