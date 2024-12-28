# Amazon RDS Setup Guide

## `config.json`

```json
{
    "user": "user",
    "password": "password",
    "host": "machine-learning-knowledge-base.c3gsw0qk00r6.us-east-2.rds.amazonaws.com",
    "port": 3306
}
```

## Create Database

```sql
SHOW DATABASES;
CREATE DATABASE `machine-learning-knowledge-base`
```

Now add `database` to `config.json`:
```json
{
    "user": "user",
    "password": "password",
    "host": "machine-learning-knowledge-base.c3gsw0qk00r6.us-east-2.rds.amazonaws.com",
    "database": "machine-learning-knowledge-base",
    "port": 3306
}
```

## Create Table

```sql
USE `machine-learning-knowledge-base`
SHOW TABLES;
CREATE TABLE IF NOT EXISTS papers (
    id VARCHAR(255) PRIMARY KEY,
    code_names JSON,
    title TEXT,
    urls JSON,
    pub_name TEXT,
    pub_date TEXT,
    authors JSON,
    abstract TEXT,
    full_text LONGTEXT,
    comments JSON
);
```
