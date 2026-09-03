# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

import gzdrl
import rclpy
from gzdrl.sitl import RosDRLServer
import time

def format_dict(dict_, ntab=0):
    if isinstance(dict_, (list, tuple)):
        fmt = ""
        for item in dict_:
            fmt += ntab*" " + str(item)+"\n"
        return fmt
    if not isinstance(dict_, dict):
        return "\t"*ntab + str(dict_)
    fmt = ""
    for key in dict_:
        fmt = ntab*"\t" + fmt + str(key) +":\n"
        fmt += format_dict(dict_[key], ntab+1)
    return fmt

if __name__ == "__main__":
    sdf_file = str(gzdrl.get_sdf_path("world_simple.sdf"))
    serv = RosDRLServer("ros", sdf_file, ["quadrotor"], True, 1.0, {"quadrotor":["quadrotor/base_link"]} )
    serv.run()
    # spin so that publishers and subscribers are activated
    # in a cli: ros2 topic list 
    # It will show all the available topics
    # Alternatively:
    print("Subscribed topics by ROSDRLServer:", format_dict(serv.get_subscribed_topic_map()))
    print("Published topics by ROSDRLServer:", format_dict(serv.get_published_topic_map()))
    serv.spin_async()
    # You can also access the underlying DRLSErver
    server = serv.server()
    # Use the DRLSErver api here!, beware of data races, because RosDRLServer is using the server inside it
    server.update_control_states()
    print("State of quadrotor: ",server.control_states["quadrotor"]["quadrotor/base_link"][0])
    # Join back to the spin thread to prevent exit
    serv.spin()
