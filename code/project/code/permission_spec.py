"""Torch-free single source of truth for the permission vocabulary (R7).

graph_featurizer / train_gnn import PERMISSION from here so that the deployment
bundle validator can compute FD without importing torch (avoids the libomp
clash with xgboost in the coordinating process).
"""
PERMISSION = [
    "EXEC",         # 0: shell_exec, code_exec, tool_execute
    "FS_READ",      # 1: file_read, database_query, information_search, data_aggregate
    "FS_CREATE",    # 2: file_write
    "FS_DELETE",    # 3: file_delete
    "NET_FETCH",    # 4: web_request, web_search, account_access
    "NET_DOWN",     # 5: web_download
    "MEM_WRITE",    # 6: memory_write, version_control
    "MEM_READ",     # 7: memory_read, memory_reflect
    "AGENT_SPAWN",  # 8: agent_spawn
    "AGENT_MSG",    # 9: agent_message
    "COMM_SEND",    # 10: email_send, message_send, social_media, document_create, prompt_inject
    "SYS_MODIFY",   # 11: system_config
]
