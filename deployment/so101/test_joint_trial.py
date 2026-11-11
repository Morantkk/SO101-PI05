"""CPU checks for the boundaries guarding real motor commands."""
import unittest
import numpy as np
from joint_trial import bounded_target, full_task_target, select_action


class TrialGuardTests(unittest.TestCase):
    def setUp(self):
        self.config = dict(policy="pi05_so101_fold", checkpoint_step=68660)
        self.message = dict(**self.config, control_mode="joint", joint_units="degrees", gripper_units="0_to_100",
                            **{"timestamp/ms": 100000, "delta/ms": 1000/30},
                            left_arm={"actions": [[n]*6 for n in range(30)]},
                            right_arm={"actions": [[n+100]*6 for n in range(30)]})

    def test_time_and_side_alignment(self):
        np.testing.assert_equal(select_action(self.message,self.config,100.15,99),[4]*6+[104]*6)

    def test_expiration_and_model_identity(self):
        for now, earliest in [(100.5,99),(99.9,99),(100.1,100.05)]:
            with self.assertRaises(ValueError): select_action(self.message,self.config,now,earliest)
        self.message['checkpoint_step']=20000
        with self.assertRaises(ValueError): select_action(self.message,self.config,100.1,99)

    def test_bad_shape_and_nan(self):
        self.message['left_arm']['actions'][0][0]=float('nan')
        with self.assertRaises(ValueError): select_action(self.message,self.config,100.1,99)
        self.message['left_arm']['actions']=[[0]*5]*30
        with self.assertRaises(ValueError): select_action(self.message,self.config,100.1,99)

    def test_speed_envelope_and_grippers(self):
        initial=np.zeros(12); initial[[5,11]]=20
        p=initial.copy(); low=np.full(12,-100); high=np.full(12,100)
        high[0]=2
        for _ in range(300):
            n=bounded_target(np.full(12,50),p,initial,p,low,high,1/30)
            self.assertLessEqual(np.max(abs(n-p)),.10000001)
            self.assertLessEqual(np.max(abs(n-initial)),5.0000001)
            self.assertLessEqual(n[0],2)
            np.testing.assert_equal(n[[5,11]],initial[[5,11]])
            p=n

    def test_tracking_stall_and_bounds(self):
        z=np.zeros(12);lo=np.full(12,-10);hi=-lo
        cases=[(z,z,z,0.2),(z,z,np.ones(12)*3,1/30),(np.ones(12)*11,z,z,1/30),
               (np.ones(12)*6,z,np.ones(12)*6,1/30)]
        for measured,initial,previous,dt in cases:
            with self.assertRaises(ValueError):bounded_target(z,measured,initial,previous,lo,hi,dt)

    def test_full_task_travel_and_grasping(self):
        p=np.zeros(12);lo=np.full(12,-90.);hi=np.full(12,90.)
        lo[[5,11]]=0;hi[[5,11]]=100
        for _ in range(100):
            n=full_task_target(np.full(12,200),p,p,lo,hi,1/30)
            self.assertLessEqual(np.max(abs(n-p)),1.52100001)
            self.assertLessEqual(np.max(abs(n[[5,11]]-p[[5,11]])),1.20000001)
            self.assertTrue(np.all(n<=hi))
            p=n
        self.assertGreater(p[0],5)
        self.assertGreater(p[5],5)

    def test_full_task_tracking_stall_and_invalid(self):
        z=np.zeros(12);lo=np.full(12,-90.);hi=-lo
        for dt in (.2,float('nan')):
            with self.assertRaises(ValueError):full_task_target(z,z,z,lo,hi,dt)
        for joint,value in [(0,9),(5,5)]:
            p=z.copy();p[joint]=value
            with self.assertRaises(ValueError):full_task_target(z,z,p,lo,hi,1/30)
        p=z.copy();p[2]=float('nan')
        with self.assertRaises(ValueError):full_task_target(p,z,z,lo,hi,1/30)

    def test_full_task_clips_to_calibration(self):
        p=np.full(12,9.9);lo=np.full(12,-10.);hi=-lo
        command=full_task_target(np.full(12,100),p,p,lo,hi,1/30)
        np.testing.assert_allclose(command,10)


if __name__=='__main__':unittest.main()
