import numpy as np
import pandas as pd
import tushare as ts
import MySQLdb as mdb

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import pandas as pd
from sklearn import cluster,covariance,manifold

from matplotlib.font_manager import FontProperties


#获取沪深300指数的股票名单
hs300_data=ts.get_hs300s()


#获取上交所SSE，深交所SZSE，港交所HKEX正常上市交易的股票名单
pro=ts.pro_api()
exc=['SSE','SZSE']
stock_data=[]
for ex in exc:
    data=pro.query('stock_basic', exchange=ex,
                 list_status='L',
                 fields='ts_code,symbol,name,area,industry,list_date')
    stock_data.append(data)


#获取沪深300成分股中正常上市交易的名单
#将stock_data中上交所和深交所中的交易数据合并
s_name=pd.concat([stock_data[0][["name","ts_code"]],
                  stock_data[1][["name","ts_code"]]],ignore_index=True)

#找出沪深300指在上交所和深交所的交易代码
hs300_data=hs300_data.set_index("name")
s_name=s_name.set_index("name")

sdata=pd.merge(hs300_data,s_name,on="name",how="inner")
ts_code=sdata["ts_code"].values

####这一部分不属于本作业，为方便以后用存储的
##存入mysql中
data=[]
for i in range(len(sdata)):
    data.append((i+1,sdata.iloc[i]["date"],sdata.iloc[i]["code"],
                 sdata.iloc[i]["weight"],
                 sdata.iloc[i]["ts_code"]))
con=mdb.connect(host='localhost',
                user='root',passwd='123456',db='lqj',
                use_unicode=True,charset='utf8')
cur=con.cursor()

sql='insert into lectr(id,date,code,weight,ts_code) values(%s,%s,%s,%s,%s)'
cur.executemany(sql,data)
con.commit()
        
#提取沪深300指2010年01月01到2018年01月01的交易数据，存入d_price中
d_price=[]
names=[]
symbols=[]
for i in range(62,199):
    df = pro.daily(ts_code=ts_code[i],
                   start_date='20181201',
                   end_date='20190101')
    d_price.append(df)
    names.append(sdata[sdata["ts_code"]==ts_code[i]].index.tolist())
    symbols.append(ts_code[i])


names=pd.DataFrame(names)
symbols=pd.DataFrame(symbols)

op=[]
cl=[]
for q in d_price:
    op.append(q['open'].values)
    cl.append(q['close'].values)

close_prices=np.vstack([i for i in op])
open_prices=np.vstack([j for j in cl])
    

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
    print('Cluster %i: %s' % ((i + 1), ', '.join(names[0][labels == i])))

# #############################################################################
# Find a low-dimension embedding for visualization: find the best position of
# the nodes (the stocks) on a 2D plane

# We use a dense eigen_solver to achieve reproducibility (arpack is
# initiated with random vectors that we don't control). In addition, we
# use a large number of neighbors to capture the large-scale structure.
node_position_model = manifold.LocallyLinearEmbedding(
    n_components=2, eigen_solver='dense', n_neighbors=10)

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

# Plot the edges，画出股票之间的连线，颜色深浅(values)表示两者之间的相关性
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
        zip(names[0], labels, embedding.T)):

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
                       alpha=.6),fontproperties=FontProperties(fname='/System/Library/Fonts/PingFang.ttc'))

plt.xlim(embedding[0].min() - .15 * embedding[0].ptp(),
         embedding[0].max() + .10 * embedding[0].ptp(),)
plt.ylim(embedding[1].min() - .03 * embedding[1].ptp(),
         embedding[1].max() + .03 * embedding[1].ptp())

plt.show()

