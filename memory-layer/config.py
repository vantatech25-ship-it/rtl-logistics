"""
RTL Logistics — Memory Layer
Manages:
  - Pinecone: vector embeddings for hub/route spatial memory
  - TimescaleDB: time-series telemetry (GPS pings, load, traffic)
"""

PINECONE_API_KEY = "your-pinecone-api-key"
PINECONE_INDEX   = "rtl-logistics-hubs"

TIMESCALE_DSN = "postgresql://rtt_user:rtt_pass@localhost:5432/rtt_logistics"
