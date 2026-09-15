# Domain Glossary

## Pending Entry

A durable AI-approved intention to open a position that has not yet passed entry revalidation and is not a broker order. Creating a Pending Entry never sends, modifies, or cancels an order.

## Entry Revalidation

The safety review performed after the regular session opens and before final risk approval. It checks current executable market data and information that became available after the original decision.

## Marketable Limit Order

A limit order priced to permit an immediate fill within a configured slippage ceiling. It is not a market order and its price is calculated by the Execution Engine, not by the AI model.

## Managed Position

A durable broker-backed position plus its entry thesis, thesis horizon, latest Sol review, and monitoring status. It is never created from an unfilled decision.

## Thesis Horizon

The expected useful life of the investment thesis. It is neither a minimum holding lock nor an automatic sell date.

## Replacement Gap

The best alternative's Sol score minus the current holding's score from one common review. Python applies the configured threshold before a REPLACE proposal can become a risk-gated SWITCH intent.
