import networkx as nx
from sklearn.preprocessing import scale
from pandas import DataFrame, Series

# courtesy of https://github.com/Diviyan-Kalainathan/

class PairwiseModel(object):
    def __init__(self):
        """Init."""
        super(PairwiseModel, self).__init__()

    def predict(self, x, *args, **kwargs):
        if len(args) > 0:
            if type(args[0]) == nx.Graph or type(args[0]) == nx.DiGraph:
                return self.orient_graph(x, *args, **kwargs)
            else:
                y = args.pop(0)
                return self.predict_proba((x, y), *args, **kwargs)
        elif type(x) == DataFrame:
            return self.predict_dataset(x, *args, **kwargs)
        elif type(x) == Series:
            return self.predict_proba((x.iloc[0], x.iloc[1]), *args, **kwargs)

    def predict_proba(self, dataset, idx=0, **kwargs):
        raise NotImplementedError

    def predict_dataset(self, x, **kwargs):
        printout = kwargs.get("printout", None)
        pred = []
        res = []
        x.columns = ["A", "B"]
        for idx, row in x.iterrows():
            a = scale(row['A'].reshape((len(row['A']), 1)))
            b = scale(row['B'].reshape((len(row['B']), 1)))

            pred.append(self.predict_proba((a, b), idx=idx))

            if printout is not None:
                res.append([row['SampleID'], pred[-1]])
                DataFrame(res, columns=['SampleID', 'Predictions']).to_csv(
                    printout, index=False)
        return pred

    def orient_graph(self, df_data, graph, printout=None, **kwargs):
        if isinstance(graph, nx.DiGraph):
            edges = [a for a in list(graph.edges()) if (a[1], a[0]) in list(graph.edges())]
            oriented_edges = [a for a in list(graph.edges()) if (a[1], a[0]) not in list(graph.edges())]
            for a in edges:
                if (a[1], a[0]) in list(graph.edges()):
                    edges.remove(a)
            output = nx.DiGraph()
            for i in oriented_edges:
                output.add_edge(*i)

        elif isinstance(graph, nx.Graph):
            edges = list(graph.edges())
            output = nx.DiGraph()

        else:
            raise TypeError("Data type not understood.")

        res = []

        for idx, (a, b) in enumerate(edges):
            weight = self.predict_proba(
                (df_data[a].values.reshape((-1, 1)),
                 df_data[b].values.reshape((-1, 1))), idx=idx,
                **kwargs)
            if weight > 0:  # a causes b
                output.add_edge(a, b, weight=weight)
            elif weight < 0:
                output.add_edge(b, a, weight=abs(weight))
            if printout is not None:
                res.append([str(a) + '-' + str(b), weight])
                DataFrame(res, columns=['SampleID', 'Predictions']).to_csv(
                    printout, index=False)

        for node in list(df_data.columns.values):
            if node not in output.nodes():
                output.add_node(node)

        return output
