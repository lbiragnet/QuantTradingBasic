#pragma once

#include <stdexcept>
#include <fmt/core.h>
#include "enums.hpp"
#include "order_book_level_info.hpp"

class Order 
{
    public:
        Order(OrderType orderType, OrderId orderId, Side side, Price price, Quantity quantity);

        OrderType GetOrderType() const;
        OrderId GetOrderId() const;
        Side GetSide() const;
        Price GetPrice() const;
        Quantity GetInitialQuantity() const;
        Quantity GetRemainingQuantity() const;
        Quantity GetFilledQuantity() const;
        void Fill(Quantity quantity);

    private:
        OrderType orderType_;
        OrderId orderId_;
        Side side_;
        Price price_;
        Quantity initialQuantity_;
        Quantity remainingQuantity_;
};