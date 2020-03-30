from zumi.zumi import Zumi
zumi = Zumi()

def line_follower(time_out,speed = 5,left_thresh=100,right_thresh=100, gain = 0.1):
    #this method will try to follow a black line on a white floor.
    #if both ir sensors detect white the Zumi will stop.
    #timeout is the amount of time you want to do line following
    #speed is base speed the motors will go forward at.
    #left thresh is the left bottom ir threshold the sensor
    #goes from 0-255 so its like a cuttoff point
    #same for right thresh but for right bottom ir sensor
    #gain is how sensitive your line following code will 
    #be to difference between the left and right ir sensors
    #if your Zumi shakes a lot left to right then try to make the gain smaller
    #if your zumi cant make sharp turns increase the gain by +0.1
    time_passed = 0
    init_time = time.time()
    try:
        while(time_passed <= time_out):
            zumi.update_angles()
            ir_readings = zumi.get_all_IR_data()
            left_bottom_ir = ir_readings[3]
            right_bottom_ir = ir_readings[1]
            
            #this is the difference between the left and right ir sensor
            #when the left value is bigger than the right then the diff will be positive
            #if the right value if bigger than the left then the diff will be negative
            diff = int((left_bottom_ir - right_bottom_ir)*gain)

            if left_bottom_ir < left_thresh and right_bottom_ir < right_thresh:
                #if both ir sensors detect white then we stop
                zumi.stop()
                break
            elif diff > 1:
                zumi.control_motors(speed+diff,0)
            elif diff < 1:
                zumi.control_motors(0,speed+abs(diff))
            else:
                #if the difference is close to 0 lets just drive forward
                zumi.control_motors(speed+diff,speed-diff)
            #update the time passed
            time_passed = time.time()-init_time
    finally:
        #always have a stop at the end of code with control motors
        zumi.stop()