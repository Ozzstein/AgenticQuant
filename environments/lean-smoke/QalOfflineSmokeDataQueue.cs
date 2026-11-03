// Test-only feed: real LEAN ticks generated locally; never a market-data authority.
using QuantConnect.Interfaces;
using QuantConnect.Data;
using QuantConnect.Util;
namespace QuantConnect.Lean.Engine.DataFeeds.Queues
{
    public class QalOfflineSmokeDataQueue : FakeDataQueue
    {
        public QalOfflineSmokeDataQueue()
            : base(Composer.Instance.GetExportedValueByTypeName<IDataAggregator>(nameof(AggregationManager)), 2) { }
    }
}
