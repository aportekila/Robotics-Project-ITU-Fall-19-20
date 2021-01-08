from baxter import *

baxter_env = BaxterEnvironment()
baxter_env.get_image()

baxter_env.prepare_table()
baxter_env.get_image()

baxter_env.run_simulation(20)

baxter_gui = Baxter(baxter_env.physics_client)
baxter_env.get_image()
baxter_back = Baxter(baxter_env.ik_back)
baxter_env.set_baxter(baxter_gui, baxter_back)

positive = 0
total = 0

while total < 100:
    orj_pos, orj_color = baxter_env.spawn_object()
    object_pos, object_color = baxter_env.detect_object()
    baxter_env.put_into_basket(object_pos, object_color)
    baxter_env.run_simulation(50)

    position = pb.getBasePositionAndOrientation(baxter_env.cube)[0]
    if orj_color != object_color:
        pass
    else:
        if position[2] > .6:
            pass
        else:
            if orj_color == "G":
                if -0.04 < position[0] < 0.34 and 0.15 < position[1] < 0.55:
                    positive += 1
            else:
                # Red
                if 0.66 < position[0] < 1.06 and 0.15 < position[1] < 0.55:
                    positive += 1
    total += 1

    print("######################################")
    print("Successful", positive)
    print("Fail", total - positive)
    print("Success Rate", str((positive * 100.0) / (total * 1.0)) + "%")
    print("Average time for 1 cube ", str(baxter_env.time_consumed / (total * 1.0)) + " s")

    pb.removeBody(baxter_env.cube)
