import numpy as np
import pandas as pd
import tushare as ts
import MySQLdb as mdb

#获取沪深300指数的股票名单
hs300_data=ts.get_hs300s()
hss=hs300_data["name"]

#获取上交所SSE，深交所SZSE，港交所HKEX正常上市交易的股票名单
pro=ts.pro_api()
exc=["SSE","SZSE"]
stock_data=[]
for ex in exc:
    data=pro.query('stock_basic', exchange=ex,
                 list_status='L',
                 fields='ts_code,symbol,name,area,industry,list_date')
    stock_data.append(data)



#获取沪深300成分股中正常上市交易的名单
s_name=pd.concat([stock_data[0][["name","ts_code"]],
                  stock_data[1][["name","ts_code"]]],ignore_index=True)


    
normal300=[]

num300=len(hss)
numn=len(s_name)

for i in range(num300):
    for j in range(numn):
        if hss.loc[i]==s_name.loc[j]["name"]:
            normal300.append(s_name.loc[i])
        
#提取沪深300指2010年01月01到2018年01月01的交易数据，存入d_price中
d_price=[]
for i in range(199): 
    df = pro.daily(ts_code=normal300[i]["ts_code"],
                   start_date='201812010',
                   end_date='20190110')
    d_price.append(df)

#获取数据后，开始聚类，即第三课第二次作业

#close_prices=pd.DataFrame()
#open_prices=pd.DataFrame()
#for q in d_price:
    #close_prices=pd.concat([close_prices,q["price"]],ignore_index=True)
    #close_prices=pd.concat([close_prices,q["price"]],ignore_index=True)
    
close_prices=np.vstack([q["close"] for q in d_price])
open_prices=np.vstack([q["open"] for q in d_price])


# The daily variations of the quotes are what carry most information
variation = close_prices - open_prices


# #############################################################################
# Learn a graphical structure from the correlations
edge_model = covariance.GraphicalLassoCV(cv=5)

# standardize the time series: using correlations rather than covariance
# is more efficient for structure recovery
X = variation.copy().T
X /= X.std(axis=0)
edge_model.fit(X)

# #############################################################################
# Cluster using affinity propagation

_, labels = cluster.affinity_propagation(edge_model.covariance_)
n_labels = labels.max()

for i in range(n_labels + 1):
    print('Cluster %i: %s' % ((i + 1), ', '.join(names[labels == i])))

# #############################################################################
# Find a low-dimension embedding for visualization: find the best position of
# the nodes (the stocks) on a 2D plane

# We use a dense eigen_solver to achieve reproducibility (arpack is
# initiated with random vectors that we don't control). In addition, we
# use a large number of neighbors to capture the large-scale structure.
node_position_model = manifold.LocallyLinearEmbedding(
    n_components=2, eigen_solver='dense', n_neighbors=6)

embedding = node_position_model.fit_transform(X.T).T

# #############################################################################
# Visualization
plt.figure(1, facecolor='w', figsize=(10, 8))
plt.clf()
ax = plt.axes([0., 0., 1., 1.])
plt.axis('off')

# Display a graph of the partial correlations
partial_correlations = edge_model.precision_.copy()
d = 1 / np.sqrt(np.diag(partial_correlations))
partial_correlations *= d
partial_correlations *= d[:, np.newaxis]
non_zero = (np.abs(np.triu(partial_correlations, k=1)) > 0.02) #np.triu返回上角矩阵

# Plot the nodes using the coordinates of our embedding
#x=embedding[0],y=embedding[1],画出代表股票的点
plt.scatter(embedding[0], embedding[1], s=100 * d ** 2, c=labels,
            cmap=plt.cm.nipy_spectral) 

# Plot the edges，画出股票之间的连线，粗细(values)表示两者之间的相关性
start_idx, end_idx = np.where(non_zero)#以元组形式返回值为true的坐标

# a sequence of (*line0*, *line1*, *line2*), where::
#            linen = (x0, y0), (x1, y1), ... (xm, ym)
segments = [[embedding[:, start], embedding[:, stop]]
            for start, stop in zip(start_idx, end_idx)]
values = np.abs(partial_correlations[non_zero]) #行列式子变换后的矩阵的非零值
lc = LineCollection(segments,
                    zorder=0, cmap=plt.cm.hot_r,
                    norm=plt.Normalize(0, .7 * values.max()))
lc.set_array(values)
lc.set_linewidths(10 * values)
ax.add_collection(lc)

# Add a label to each node. The challenge here is that we want to
# position the labels to avoid overlap with other labels
#x,y为表示股票坐标的点的坐标
for index, (name, label, (x, y)) in enumerate(
        zip(names, labels, embedding.T)):

    dx = x - embedding[0] #某一个点的横坐标于其他所有点的横坐标的差值，长度为56
    dx[index] = 1 #设第index个值为1，原值为0
    dy = y - embedding[1]#某一个点的纵坐标于其他所有点的纵坐标的差值，长度为56
    dy[index] = 1#设第index个值为1，原值为0
    this_dx = dx[np.argmin(np.abs(dy))]
    #np.argmin()返回最小值所在的下标,本语句为求出dy绝对值最小值所在的dx坐标
    this_dy = dy[np.argmin(np.abs(dx))]
    #np.argmin()返回最小值所在的下标,本语句为求出dy绝对值最小值所在的dx坐标
    if this_dx > 0:
        horizontalalignment = 'left'
        x = x + .002
    else:
        horizontalalignment = 'right' 
        x = x - .002
    if this_dy > 0:
        verticalalignment = 'bottom'
        y = y + .002
    else:
        verticalalignment = 'top'
        y = y - .002
    plt.text(x, y, name, size=10,
             horizontalalignment=horizontalalignment,
             verticalalignment=verticalalignment,
             bbox=dict(facecolor='w',
                       edgecolor=plt.cm.nipy_spectral(label / float(n_labels)),
                       alpha=.6))

plt.xlim(embedding[0].min() - .15 * embedding[0].ptp(),
         embedding[0].max() + .10 * embedding[0].ptp(),)
plt.ylim(embedding[1].min() - .03 * embedding[1].ptp(),
         embedding[1].max() + .03 * embedding[1].ptp())

plt.show()


   




    
    
    


