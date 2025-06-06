#pragma once

#include <memory>
#include <list>
#include "order.hpp"
#include "enums.hpp"
#include "order_book_level_info.hpp"

using OrderPointer = std::shared_ptr<Order>;
using OrderPointers = std::list<OrderPointer>;

class OrderModify
{
    public:
        OrderModify(OrderId orderId, Side side, Price price, Quantity quantity);

        OrderId GetOrderId() const;
        Price GetPrice() const;
        Side GetSide() const;
        Quantity GetQuantity() const;

        OrderPointer ToOrderPointer(OrderType type) const;

    private:
        OrderId orderId_;
        Price price_;
        Side side_;
        Quantity quantity_;
};


