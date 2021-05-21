def save_best_checkpoint(self, runner, key_score):
        best_score = runner.meta['hook_msgs'].get(
            'best_score', self.init_value_map[self.rule])
        if self.compare_func(key_score, best_score):
            best_score = key_score
            runner.meta['hook_msgs']['best_score'] = best_score
            last_ckpt = runner.meta['hook_msgs']['last_ckpt']
            runner.meta['hook_msgs']['best_ckpt'] = last_ckpt
            # mmcv.symlink(
            #     last_ckpt,
            #     osp.join(runner.work_dir, f'best_{self.key_indicator}.pth'))
            shutil.copy(
                last_ckpt,
                osp.join(runner.work_dir, f'best_{self.key_indicator}.pth')
            )
            
            time_stamp = runner.epoch + 1 if self.by_epoch else runner.iter + 1
            self.logger.info(f'Now best checkpoint is epoch_{time_stamp}.pth.'
                             f'Best {self.key_indicator} is {best_score:0.4f}')
        os.remove(osp.join(runner.work_dir,f'epoch_{runner.epoch + 1}.pth'))