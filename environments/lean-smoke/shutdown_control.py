from AlgorithmImports import QCAlgorithm, Resolution


class SmokeAlgorithm(QCAlgorithm):
    def Initialize(self):
        self.AddCrypto("BTCUSD", Resolution.Second, "coinbase")

    def OnData(self, data):
        self.Quit("control: no project execution bridge, no order")
