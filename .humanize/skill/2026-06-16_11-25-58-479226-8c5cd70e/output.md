CONVERGED: no
REMAINING_REQUIRED_CHANGES:
- AC-0.3/task6 still uses the wrong capacity path: change `/server_info` `internal_states[*].token_capacity` to `internal_states[*].memory_usage.token_capacity`, and explicitly require/capture `internal_states[*].effective_max_running_requests_per_dp`.
NEW_RISKS:
- none
DEC_SET_OK: yes
