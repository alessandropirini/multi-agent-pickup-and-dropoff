# Multi-Agent Pickup and Dropoff for Makespan Optimization

A planning-and-control stack for a fleet of differential-drive robots that must collect scattered items and deliver them to designated collection zones in a warehouse-like map. The system uses a one-time global planning phase to coordinate the team (goal assignment and high-level routes), then runs fully decentralized execution: each robot acts only on its local LiDAR-based observations and treats other robots as dynamic obstacles. The objective is to minimize the overall completion time (makespan / task accomplishment time) while avoiding collisions and operating within real-time control constraints.

