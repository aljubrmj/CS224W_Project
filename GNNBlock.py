class GNNBlock(torch.nn.Module):
    def __init__(self, num_nodes, in_channels, out_channels, hidden_size, last_step=False, add_self_loops=False):
        super(GNNBlock, self).__init__()

        self.last_step = last_step
        self.gcn = GCNConv(in_channels, out_channels, add_self_loops=add_self_loops)

        self.linearf1 = Linear(in_channels, 2*in_channels)
        self.linearf2 = Linear(in_channels, 2*in_channels)
        self.lineara1 = Linear(2*in_channels, hidden_size)
        self.bna1 = BatchNorm2d(num_nodes)
        self.lineara2 = Linear(hidden_size, hidden_size)
        self.bna2 = BatchNorm2d(num_nodes)
        self.lineara_last = Linear(hidden_size, 1)
        self.bna_last = BatchNorm2d(num_nodes)
        self.bnz = BatchNorm1d(num_nodes)
        self.linearz = Linear(out_channels, out_channels)

        self.lrelu = LeakyReLU(0.1)
        self.dropout = Dropout(0.1)

        self.adj1 = torch.rand((num_nodes, num_nodes)).to(device)
        self.edge_index = (self.adj1 > 0).nonzero().t()

    def forward(self, x):
        f1 = self.lrelu(self.linearf1(torch.abs(x.unsqueeze(2) - x.unsqueeze(1))))
        f2 = self.lrelu(self.linearf2(torch.mul(x.unsqueeze(2), x.unsqueeze(1))))
        A_hat = f1 + f2
        A_hat = self.lrelu(self.bna1(self.lineara1(A_hat)))
        A_hat = self.dropout(self.lrelu(self.bna2(self.lineara2(A_hat))))
        A_hat = torch.mean(self.lineara_last(A_hat), dim=0).squeeze()
        A_hat = torch.sigmoid((A_hat + A_hat.t())/2)

        if self.last_step:
          pred = self.gcn(x, self.edge_index, torch.flatten(A_hat))
        else:
          pred = self.lrelu(self.bnz(self.gcn(x, self.edge_index, torch.flatten(A_hat))))

        return pred, A_hat
