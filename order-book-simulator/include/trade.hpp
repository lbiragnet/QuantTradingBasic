#pragma once

#include "enums.hpp"
#include "order_book_level_info.hpp"

struct TradeInfo
{
    OrderId orderId_;
    Price price_;
    Quantity quantity_;
};

class Trade
{
    public:
        Trade(const TradeInfo& bidTrade, const TradeInfo& askTrade);

    private:
        TradeInfo bidTrade_;
        TradeInfo askTrade_;
};