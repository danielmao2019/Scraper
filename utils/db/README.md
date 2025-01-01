# Amazon RDS Setup Guide

## Add Inbound Rule

## `config.json`

```json
{
    "user": "user",
    "password": "password",
    "host": "machine-learning-knowledge-base.c3gsw0qk00r6.us-east-2.rds.amazonaws.com",
    "dbname": "postgres",
    "port": 5432
}
```

## Create Database

```sql
CREATE DATABASE "machine-learning-knowledge-base"
```

Verify
```sql
SELECT datname FROM pg_database WHERE datistemplate = false;
```

Now add `database` to `config.json`:
```json
{
    "user": "user",
    "password": "password",
    "host": "machine-learning-knowledge-base.c3gsw0qk00r6.us-east-2.rds.amazonaws.com",
    "dbname": "machine-learning-knowledge-base",
    "port": 5432
}
```

## Create Table

```sql
CREATE TABLE IF NOT EXISTS papers (
    id TEXT,
    code_names TEXT[],
    title TEXT PRIMARY KEY,
    urls JSONB,
    pub_name TEXT,
    pub_date TEXT,
    authors TEXT[],
    abstract TEXT,
    full_text TEXT,
    comments TEXT[]
);
```

Verity
```sql
SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';
```

## Future Work

1. Batched processing.
