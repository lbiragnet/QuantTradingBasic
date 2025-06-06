#pragma once

#include <cstdint>;
#include <vector>
using Price = std::int32_t;
using Quantity = std::uint32_t;
using OrderId = std::uint64_t;
using LevelInfos = std::vector<LevelInfo>;

struct LevelInfo {
    Price price_;
    Quantity quantity_;
};

class OrderbookLevelInfos
{
    public:
        OrderbookLevelInfos(const LevelInfos& bids, const LevelInfos& asks);

    private:
        LevelInfos bids_;
        LevelInfos asks_;
};
