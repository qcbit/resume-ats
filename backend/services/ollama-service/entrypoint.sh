#!/bin/sh
ollama serve &
sleep 2
ollama pull llama2:7b-chat
wait